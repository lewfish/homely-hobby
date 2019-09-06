#!/usr/bin/env python3

import boto3
import datetime

client = boto3.client('ec2', region_name='us-east-1')

response = client.request_spot_instances(
    DryRun=False,
    SpotPrice='0.5',
    InstanceCount=1,
    Type='one-time',
    LaunchSpecification={
        'ImageId': 'ami-0241ac1f637a90b84',
        'KeyName': 'lewfish-raster-vision',
        'InstanceType': 'p2.xlarge',
        'Placement': {
            'AvailabilityZone': 'us-east-1a',
        },
        'SecurityGroupIds': [
            'sg-06984b3d26ba2115e'
        ],
        'BlockDeviceMappings': [
            {
                'DeviceName': '/dev/xvda',
                'Ebs': {
                    'VolumeSize': 100,
                    'DeleteOnTermination': True,
                    'VolumeType': 'gp2',
                    'Encrypted': False
                },
            },
        ],
        'EbsOptimized': False,
        'Monitoring': {
            'Enabled': True
        },
    }
)

print(response)
print()
instances = client.instances.filter(
    Filters=[{'Name': 'instance-state-name', 'Values': ['running']}])
for instance in instances:
    print("Id: {}, type: {}, ip: {}".format(instance.id, instance.instance_type, instance.public_ip_address))