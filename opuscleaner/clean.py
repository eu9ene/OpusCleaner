#!/usr/bin/env python3
"""Stand-alone filter pipeline runner. Executes all of the filters defined in
a dataset filtering pipeline created by empty-train in their own process and
links them together through pipes. Can read from stdin but by default reads
the dataset from the same folder as the pipeline configuration file.
"""
import argparse
import json
import os
import shlex
import signal
import sys
import traceback
from contextlib import ExitStack
from glob import glob
from pprint import pprint
from queue import Queue, SimpleQueue
from shlex import quote
from shutil import copyfileobj
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Thread
from typing import Dict, List, Any, BinaryIO, TextIO, Optional, TypeVar, Iterable, Tuple, NamedTuple, Union
from io import TextIOWrapper

from pydantic import parse_obj_as

from opuscleaner import logging
from opuscleaner.config import COL_PY, FILTER_PATH
from opuscleaner.filters import list_filters, set_global_filters, filter_format_command, Filter, FilterStep, FilterPipeline, quote, format_shell
from opuscleaner._util import none_throws, ThreadPool, CancelableQueue, Cancelled


# Queue for printing lines to stdout or stderr. None means end of input.
PrintQueue = SimpleQueue[Union[None,bytes]]

# Control queue for communicating the return code of a child back to the parent.
ControlQueue = SimpleQueue[Tuple[int,int]]

# Batches to be processed. tuple[batch index,batch path]. None means end of input.
# Using a Queue here to limit the maximum capacity.
BatchQueue = CancelableQueue[Union[None,Tuple[int,str]]]

# Batches to be merged. Same format as BatchQueue
MergeQueue = CancelableQueue[Union[None,Tuple[int,str]]]


@logging.trace
def babysit_child(n: int, child: Popen, name: str, print_queue: PrintQueue, ctrl_queue: ControlQueue) -> None:
    """Thread that looks after a child process and passes (and prefixes) all of
    its stderr to a queue. It will tell the parent thread about the end of the
    child through the ctrl_queue.
    """
    logging.update(pid=child.pid)

    prefix = f'[{name}] '.encode()

    for line in none_throws(child.stderr):
        print_queue.put(prefix + line)

    child.wait()

    logging.event('child_exited', n=n, retval=child.returncode)

    ctrl_queue.put((n, child.returncode))


def print_lines(queue: PrintQueue, fout: BinaryIO) -> None:
    """Thread that prints stderr lines from all the children to stderr in an
    orderly fashion.
    """
    while True:
        line = queue.get()
        if line is None:
            break
        fout.write(line)

        # Since we're writing stderr, we flush after each line to make it more
        # useful for debugging
        fout.flush()


T = TypeVar('T')

def mark_last(iterable: Iterable[T]) -> Iterable[Tuple[bool,T]]:
    it = iter(iterable)
    curr_el = next(it)
    while True:
        try:
            next_el = next(it)
            yield False, curr_el
            curr_el = next_el
        except StopIteration:
            break
    yield True, curr_el



class Child(NamedTuple):
    name: str
    process: Popen
    babysitter: Thread

@logging.trace_context
class ProcessPool:
    """Context manager for spawning and babysitting child processes that are
    siblings connected by their pipes.
    """
    print_prefix: str

    ctrl_queue: ControlQueue

    print_queue: PrintQueue

    environ: Dict[str,str]

    children: List[Child]

    def __init__(self, print_queue: PrintQueue, *, env:Dict[str,str]={}, print_prefix:str=''):
        self.print_prefix = print_prefix
        self.ctrl_queue = SimpleQueue()
        self.print_queue = print_queue
        self.environ = dict(env)
        self.children = []

    def start(self, name:str, cmd: Union[str,List[str]], **kwargs) -> Popen:
        child = Popen(cmd, **{
            **kwargs,
            'env': {
                **os.environ,
                **self.environ,
                **(kwargs.get('env') or dict())
            }
        })
        n = len(self.children)
        thread = Thread(target=babysit_child, args=[n, child, name, self.print_queue, self.ctrl_queue])
        thread.start()
        self.children.append(Child(name, child, thread))
        return child

    def __enter__(self) -> 'ProcessPool':
        return self

    def __exit__(self, err_type, err_inst, err_trace) -> None:
        # Wait for the children to exit, and depending on their retval exit early
        running_children = len(self.children)

        # If we hit __exit__ due to an exception, just terminate everything
        if err_type:
            for child in self.children:
                child.process.terminate()

        # Place to store a non-zero child exit
        problem_child: Optional[Child] = None

        # Wait for all children to signal their exit
        try:
            while running_children > 0:
                child_i, retval = self.ctrl_queue.get()
                running_children -= 1

                logging.event('child_stopped', child_i=child_i, retval=retval)

                # Early exit when a process errored out. SIGPIPE is retuned by
                # processes that can no longer write to the next one. E.g. when
                # `head -n 10` stops reading because it has read its 10 lines.
                if retval not in (0, -signal.SIGPIPE):
                    problem_child = self.children[child_i]
                    break
        except KeyboardInterrupt:
            # Oh user doesn't want to wait? Okay, then we terminate.
            for child in self.children:
                child.process.terminate()
            pass

        # Wait for all the processes to exit to prevent zombies
        for child in self.children:
            if child.process.returncode is None:
                child.process.wait()

        # Wait for the babysitters to exit, which happens when their process has stopped
        for child in self.children:
            child.babysitter.join()

        # If we broke out of our ctrl_queue loop we did so because there was an issue
        # with one of the children. Let's raise that to the parent.
        if not err_inst and problem_child:
            raise RuntimeError(f"Child {problem_child.name} (pid {problem_child.process.pid}) exited with {problem_child.process.returncode}")


class PipelineStep(NamedTuple):
    name: str
    command: str
    basedir: str


class Pipeline:
    def __init__(self, filters:Dict[str,Filter], languages: List[str], pipeline: FilterPipeline):
        self.steps: List[PipelineStep] = []

        # Make sure the path to the python binary (and the installed utils)
        # is in the PATH variable. If you load a virtualenv this happens by
        # default, but if you call it with the virtualenv's python binary 
        # directly it wont.
        pyenv_bin_path = os.path.dirname(sys.executable)
        os_env_bin_paths = os.environ.get('PATH', '').split(os.pathsep)
        self.env: Optional[Dict[str,str]] = {
            **os.environ,
            'PATH': os.pathsep.join([pyenv_bin_path] + os_env_bin_paths)
        } if pyenv_bin_path not in os_env_bin_paths else None

        # Assert we have all filters we need
        assert set(step.filter for step in pipeline.filters) - set(filters.keys()) == set()

        # Make sure the path to the python binary (and the installed utils)
        # is in the PATH variable. If you load a virtualenv this happens by
        # default, but if you call it with the virtualenv's python binary 
        # directly it wont.
        pyenv_bin_path = os.path.dirname(sys.executable)
        os_env_bin_paths = os.environ.get('PATH', '').split(os.pathsep)
        self.env: Optional[Dict[str,str]] = {
            **os.environ,
            'PATH': os.pathsep.join([pyenv_bin_path] + os_env_bin_paths)
        } if pyenv_bin_path not in os_env_bin_paths else None

        for step in pipeline.filters:
            filter_def = filters[step.filter]
            command_str = filter_format_command(filter_def, step, languages)
            self.steps.append(PipelineStep(step.filter, command_str, filter_def.basedir))

    def run(self, pool:ProcessPool, stdin:BinaryIO, stdout:BinaryIO, *, tee:bool=False, basename:str="") -> None:
        if not self.steps:
            copyfileobj(stdin, stdout)
            return

        for i, (is_last_step, step) in enumerate(mark_last(self.steps)):
            child = pool.start(f'{pool.print_prefix}{i}:{step.name}', step.command,
                stdin=stdin,
                stdout=stdout if is_last_step and not tee else PIPE,
                stderr=PIPE,
                cwd=step.basedir,
                env=self.env,
                shell=True)

            # Close our reference to the previous child, now taken over by the next child
            stdin.close()
            
            # Set stdin for next step (unless there is none, then child.stdout is None)
            if not is_last_step and not tee:
                stdin = none_throws(child.stdout)

            # If we are tee-ing for debug, shunt the output to a separate file
            # TODO: uncompressed at the moment. Might be trouble.
            if tee:
                tee_child = pool.start(f'tee {i}',
                    ['tee', f'{basename}.step-{i}.tsv'],
                    stdin=stdin,
                    stdout=stdout if is_last_step else PIPE,
                    stderr=PIPE)

                stdin.close()
                stdin = none_throws(tee_child.stdout)

    def dump(self, out:TextIO) -> None:
        if self.env:
            for key, val in self.env:
                out.write(f'export {key}={quote(format_shell(val))}\n')

        for i, (is_last_step, step) in enumerate(mark_last(self.steps)):
            out.write(f'(cd {quote(format_shell(step.basedir))} && ({step.command}))')
            out.write('\n' if is_last_step else ' |\n')



def split_input(print_queue:PrintQueue, parallel: int, batch_queue: BatchQueue, batch_size:int, stdin:BinaryIO) -> None:
    """Reads data from `stdin` and splits it into chunks of `batch_size` lines.
    These chunks are stored in temporary files, whose filenames are put onto
    `batch_queue`.
    """
    more = True

    batch_index = 0

    while more:
        fh = NamedTemporaryFile(delete=False)
        
        lines = 0

        while lines < batch_size:
            line = stdin.readline()
            if line == b'':
                more = False
                break
            fh.write(line)
            lines += 1
        
        fh.close()

        try:    
            if lines > 0:
                batch_queue.put((batch_index, fh.name))
            else:
                # Empty chunk because `len(stdin) % batch_size == 0`. No need
                # to process it further.
                os.unlink(fh.name)
        except Cancelled:
            # batch_queue got interrupted, so fn.name never made it into the
            # queue. Let's clean that up.
            os.unlink(fh.name)
            raise

        batch_index += 1

    # Tell all the runners there will be no more batches coming.
    for _ in range(parallel):
        batch_queue.put(None)


@logging.trace
def run_pipeline(print_queue:PrintQueue, batch_queue:BatchQueue, merge_queue:MergeQueue, pipeline:Pipeline) -> None:
    """Receives an input filename from `batch_queue`, and once that has been processed
    with `pipeline`, it will post the output filename to `merge_queue`.

    TODO: This could also instead run ./run.py on the input and output files
    directly as opposed to using `ProcessPool` + `pipeline.run()`.

    TODO: We can rewrite this to call `srun` on SLUM clusters so that the
    actual filtering pipeline is executed on a different node. Since input
    and output are just files on the same filesystem (depends on TMPDIR) this
    should pretty much work out of the box :O
    """
    with TemporaryDirectory() as tmpdir:
        while True:
            entry = batch_queue.get()

            # If the batcher told us they're out of batches, stop.
            if entry is None:
                break

            batch_index, filename = entry

            try:
                # Write pipeline output to tempfile that is then passed on to merger.
                stdout = NamedTemporaryFile(delete=False)

                # Open chunk file and process pool and run the pipeline with it.
                # The pool's __exit__() will make us wait till the pipeline is done.
                with logging.span('run_pipeline_batch', batch_index=batch_index), \
                    open(filename, 'rb') as stdin, \
                    ProcessPool(print_queue, env={'TMPDIR': tmpdir}, print_prefix=f'{batch_index}/') as pool:
                    pipeline.run(pool, stdin, stdout)

                stdout.close()

                # Tell merger that they can process this batch when the time comes
                merge_queue.put((batch_index, stdout.name))
            except Exception as exc:
                # Didn't get to put it on the queue, delete it.
                os.unlink(stdout.name)
                
                # Add a bit more info, and re-raise
                raise RuntimeError(f'Error while processing batch {batch_index}') from exc
            finally:
                # Delete the input file from disk.
                os.unlink(filename)
        
        # Tell the merger that they should not be expecting more input from you.
        merge_queue.put(None)


def merge_output(print_queue:PrintQueue, parallel:int, merge_queue:MergeQueue, stdout:BinaryIO) -> None:
    """Takes batch filenames and numbers from `merge_queue` and will concatenate
    files in the order of the batches. If batches arrive out of order, it will
    wait for the next in order batch to arrive before continuing to concatenate.
    """
    next_batch_index = 0

    pending_batches: Dict[int, str] = {}

    while True:
        # If we have the next batch, start processing it into the final output
        if next_batch_index in pending_batches:
            batch_index, filename = next_batch_index, pending_batches[next_batch_index]

            try:
                with logging.span(f'merge_output_batch', batch_index=batch_index), open(filename, 'rb') as fh:
                    copyfileobj(fh, stdout)
            except Exception as exc:
                raise RuntimeError(f'Error while merging batch {batch_index}') from exc
            finally:
                os.unlink(filename)

            next_batch_index += 1
        # If not yet, we wait on the queue to come through with (hopefully) the next batch
        elif parallel > 0:
            entry = merge_queue.get()

            # Another batch processor finished
            if entry is None:
                parallel -= 1
            else:
                batch_index, filename = entry
                assert batch_index not in pending_batches
                pending_batches[batch_index] = filename
        # next_batch_index is not in pending_batches, and there are no more batches coming from
        # any batch processors. So let's stop.
        else:
            break

    if len(pending_batches) and next_batch_index <= max(pending_batches.keys()):
        raise RuntimeError(f'Not all batches got merged: {next_batch_index=} <= {max(pending_batches.keys())=}')

@logging.trace
def run_parallel(pipeline:Pipeline, stdin:BinaryIO, stdout:BinaryIO, *, parallel:int, batch_size:int, print_queue: PrintQueue) -> None:
    batch_queue: BatchQueue = CancelableQueue(parallel * 2)

    merge_queue: MergeQueue = CancelableQueue()

    with ThreadPool() as pool:
        # Splits stdin into files of `batch_size` lines, and puts those on `batch_queue`
        pool.start(split_input, print_queue, parallel, batch_queue, batch_size, stdin)

        # Read `batch_queue` for batch filenames, and process them. Put output files
        # on `merge_queue`.
        for n in range(parallel):
            pool.start(run_pipeline, print_queue, batch_queue, merge_queue, pipeline)

        # Read from `merge_queue` and combine files in order.
        pool.start(merge_output, print_queue, parallel, merge_queue, stdout)

        try:
            pool.join()
        except BaseException as exc: # Note: also catches KeyboardInterrupt
            batch_queue.cancel()
            merge_queue.cancel()
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--filters', '-f', type=str, default=FILTER_PATH, help='Path to directory with filter specifications')
    parser.add_argument('--input', '-i', type=argparse.FileType('rb'), help='Input tsv. If unspecified input files are read from filter json; use - to read from stdin')
    parser.add_argument('--output', '-o', type=argparse.FileType('wb'), default=sys.stdout.buffer, help='Output tsv (defaults to stdout)')
    parser.add_argument('--basedir', '-b', type=str, help='Directory to look for data files when --input is not used (defaults to same as input pipeline file)')
    parser.add_argument('--tee', action='store_true', help='Write output after each step to a separate file')
    parser.add_argument('--parallel', type=int, default=1, help='Run N parallel copies of the pipeline processing batches')
    parser.add_argument('--batch-size', type=int, default=1_000_000, help='Batch size in lines that each parallel copy processes (only if --parallel > 1)')
    parser.add_argument('--first', type=int, default=0, help='Limit reading input to the N first lines')
    parser.add_argument('--dump', action='store_true', help='Print shell script instead')
    parser.add_argument('--trace', type=argparse.FileType('w'), help='Write tracing JSON to file')
    parser.add_argument('pipeline', metavar='PIPELINE', type=argparse.FileType('r'), help='Pipeline steps specification file, e.g. *.filters.json')
    parser.add_argument('languages', metavar='LANG', type=str, nargs='*', help='Language codes of the columns in the input TSV. Only used when --input is set')

    args = parser.parse_args()

    with logging.Context(file=args.trace), logging.span('main'):
        # default search path for the data files is next to the configuration file
        # which is the default save location for empty-train.
        if not args.basedir:
            args.basedir = os.path.dirname(args.pipeline.name) or os.getcwd()

        if args.input is not None and not args.languages:
            parser.error('When --input is specified, each column\'s LANG has to be specified as well.')

        # load all filter definitions (we need to, to get their name)
        filters = {
            definition.name: definition
            for definition in list_filters(args.filters)
        }

        # set_global_filters() provides the filters to the validators in FilterPipeline
        set_global_filters(filters)
        pipeline_config = parse_obj_as(FilterPipeline, json.load(args.pipeline))

        # Order of columns. Matches datasets.py:list_datasets(path)
        languages: List[str] = args.languages if args.input else [filename.rsplit('.', 2)[1] for filename in pipeline_config.files]

        # Directory plus basename to write debug (`--tee`) files to
        basename: str = 'stdin' if args.input else os.path.commonprefix(pipeline_config.files).rstrip('.')

        pipeline = Pipeline(filters, languages, pipeline_config)

        # Input for next child
        stdin: BinaryIO

        # Output of this program
        stdout:BinaryIO = args.output

        # If we're just dumping the pipeline, do so to the specified output
        if args.dump:
            pipeline.dump(TextIOWrapper(stdout))
            sys.exit(0)

        # Queue filled by the babysitters with the stderr of the children, consumed
        # by `print_lines()` to prevent racing on stderr.
        print_queue = SimpleQueue() # type: SimpleQueue[Optional[bytes]]

        # First start the print thread so that we get immediate feedback from the
        # children even if all of them haven't started yet.
        print_thread = Thread(target=print_lines, args=[print_queue, sys.stderr.buffer])
        print_thread.start()

        # Start child processes, each reading the output from the previous sibling
        try:
            with ProcessPool(print_queue) as pool:
                # If we're not reading from stdin, read from files and paste them together
                if args.input:
                    stdin = args.input
                else:
                    # Open `gzunip` for each language file
                    gunzips = [
                        pool.start(f'gunzip {filename}',
                            ['gzip', '-cd', filename],
                            stdout=PIPE,
                            stderr=PIPE,
                            cwd=args.basedir)
                        for filename in pipeline_config.files
                    ]

                    fds = [none_throws(gunzip.stdout).fileno() for gunzip in gunzips]

                    # .. and a `paste` to combine them into columns
                    paste = pool.start('paste',
                        ['paste'] + [f'/dev/fd/{fd}' for fd in fds],
                        stdout=PIPE,
                        stderr=PIPE,
                        pass_fds=fds)

                    # Now that `paste` has inherited all the children, close our connection to them
                    for gunzip in gunzips:
                        none_throws(gunzip.stdout).close()

                    stdin = none_throws(paste.stdout)

                # If we only want the first N lines processed, use `head` to chop those off.
                if args.first > 0:
                    head = pool.start('head',
                        ['head', '-n', str(args.first)],
                        stdin=stdin,
                        stdout=PIPE,
                        stderr=PIPE)

                    stdin.close() # now taken over by `head`.
                    stdin = none_throws(head.stdout)

                if args.parallel > 1:
                    run_parallel(pipeline, stdin, stdout, print_queue=print_queue, parallel=args.parallel, batch_size=args.batch_size)
                else:
                    pipeline.run(pool, stdin, stdout, tee=args.tee, basename=basename)
        except:
            # If we didn't cleanly exit all processes, we err as well
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
        finally:
            # Tell print thread to stop (there are no more babysitters now to produce printable stuff)
            print_queue.put(None)
            print_thread.join()


if __name__ == '__main__':
    main()
