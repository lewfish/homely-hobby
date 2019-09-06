#!/usr/bin/env python3

import boto3
import datetime

client = boto3.client('ec2', region_name='us-east-1')

response = client.request_spot_fleet(
    DryRun=False,
    SpotFleetRequestConfig={
        'IamFleetRole': 'arn:aws:iam::279682201306:role/lfishgoldRasterVisionSpotFleetRole',
        'LaunchSpecifications': [
            {
                'ImageId': 'ami-0241ac1f637a90b84',
                'KeyName': 'raster-vision-team',
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
        ],

    }
    SpotPrice='0.5',
    TargetCapacity=1,
    Type='request'
)

spot_instance_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']

print('Waiting for spot request to be fulfilled...')
waiter = client.get_waiter('spot_instance_request_fulfilled')
waiter.wait(SpotInstanceRequestIds=[spot_instance_request_id])
print('Spot request fulfilled!')

instance_id = client.describe_spot_instance_requests(
    SpotInstanceRequestIds=[spot_instance_request_id])['SpotInstanceRequests'][0]['InstanceId']
public_dns = client.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]['PublicDnsName']

print('Waiting for instance to be running...')
waiter = client.get_waiter('instance_running')
waiter.wait(InstanceIds=[instance_id])
print('Instance is running!')

print()
print('ssh -i ~/.aws/raster-vision-team.pem ec2-user@{}'.format(public_dns))