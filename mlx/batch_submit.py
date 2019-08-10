#!/usr/bin/env python3

import uuid

import click

job_def = 'lewfishPyTorchCustomGpuJobDefinition'
job_queue = 'lewfishRasterVisionGpuJobQueue'

@click.command()
@click.argument('cmd')
@click.option('--debug', is_flag=True)
@click.option('--profile', is_flag=True)
@click.option('--attempts', default=5)
def batch_submit(cmd, debug, profile, attempts):
    import boto3
    client = boto3.client('batch')
    job_name = 'mlx-{}'.format(uuid.uuid4())

    cmd_list = cmd.split(' ')
    if debug:
        cmd_list = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m'] + cmd_list

    if profile:
        cmd_list = ['kernprof', '-v', '-l'] + cmd_list

    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_def,
        'containerOverrides': {
            'command': cmd_list
        },
        'retryStrategy': {
            'attempts': attempts
        }
    }

    job_id = client.submit_job(**kwargs)['jobId']
    msg = 'submitted job with jobName={} and jobId={}'.format(
        job_name, job_id)
    print(cmd_list)
    print(msg)
    return job_id

if __name__ == '__main__':
    batch_submit()