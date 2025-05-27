import pulumi
import pulumi_aws as aws
import os

instance_type = 't2.micro'
ami = 'ami-01811d4912b4ccb26'
key_name = 'cluster-key'

# Create a VPC
vpc = aws.ec2.Vpc(
    'cluster-vpc',
    cidr_block='10.0.0.0/16',
    enable_dns_support=True,
    enable_dns_hostnames=True,
    tags={'Name': 'cluster-vpc'}
)

# Create a subnet
subnet = aws.ec2.Subnet(
    'cluster-subnet',
    vpc_id=vpc.id,
    cidr_block='10.0.1.0/24',
    availability_zone='ap-southeast-1a',
    map_public_ip_on_launch=True,
    tags={'Name': 'cluster-subnet'}
)

# Create an Internet Gateway
internet_gateway = aws.ec2.InternetGateway(
    'cluster-internet-gateway',
    vpc_id=vpc.id,
    tags={'Name': 'cluster-internet-gateway'}
)

# Create a Route Table
route_table = aws.ec2.RouteTable(
    'cluster-route-table',
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block='0.0.0.0/0',
            gateway_id=internet_gateway.id,
        )
    ],
    tags={'Name': 'cluster-route-table'}
)

# Associate the route table with the subnet
route_table_association = aws.ec2.RouteTableAssociation(
    'cluster-route-table-association',
    subnet_id=subnet.id,
    route_table_id=route_table.id
)

# Create a security group with egress and ingress rules
security_group = aws.ec2.SecurityGroup(
    'cluster-security-group',
    vpc_id=vpc.id,
    description="Cluster security group",
    ingress=[
        # SSH access from anywhere
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=22,
            to_port=22,
            cidr_blocks=['0.0.0.0/0'],
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol='-1',
            from_port=0,
            to_port=0,
            self=True,
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol='-1',
            from_port=0,
            to_port=0,
            cidr_blocks=['0.0.0.0/0'],
        )
    ],
    tags={
        'Name': 'cluster-security-group'
    }
)


# Create EC2 Instances
instances = []
for i in range(2):
    instance = aws.ec2.Instance(
        f'instance-{i}',
        instance_type=instance_type,
        ami=ami,
        subnet_id=subnet.id,
        key_name=key_name,
        vpc_security_group_ids=[security_group.id],
        associate_public_ip_address=True,
        tags={
            'Name': f'instance-{i}'
        }
    )
    instances.append(instance)


pulumi.export('instance_ips', [instance.public_ip for instance in instances])
pulumi.export('instance_ids', [instance.id for instance in instances])
pulumi.export('instance_private_ips', [instance.private_ip for instance in instances])
