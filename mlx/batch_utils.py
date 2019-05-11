def submit_job(job_name, job_def, job_queue, cmd_list):
    import boto3
    client = boto3.client('batch')

    attempts = 1

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
    print(msg)
    return job_id
