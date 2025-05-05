import pulumi
import pulumi_aws as aws
import os

# Create a VPC
vpc = aws.ec2.Vpc(
    'db-cluster-vpc',
    cidr_block='10.0.0.0/16',
    enable_dns_support=True,
    enable_dns_hostnames=True,
    tags={'Name': 'db-cluster-vpc'}
)

# Create a subnet
subnet = aws.ec2.Subnet(
    'db-cluster-subnet',
    vpc_id=vpc.id,
    cidr_block='10.0.1.0/24',
    map_public_ip_on_launch=True,
    tags={'Name': 'db-cluster-subnet'}
)

# Create an Internet Gateway
internet_gateway = aws.ec2.InternetGateway(
    'db-cluster-internet-gateway',
    vpc_id=vpc.id,
    tags={'Name': 'db-cluster-internet-gateway'}
)

# Create a Route Table
route_table = aws.ec2.RouteTable(
    'db-cluster-route-table',
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block='0.0.0.0/0',
            gateway_id=internet_gateway.id,
        )
    ],
    tags={'Name': 'db-cluster-route-table'}
)

# Associate the route table with the subnet
route_table_association = aws.ec2.RouteTableAssociation(
    'db-cluster-route-table-association',
    subnet_id=subnet.id,
    route_table_id=route_table.id
)

# Create a security group with egress and ingress rules
security_group = aws.ec2.SecurityGroup(
    'db-cluster-security-group',
    vpc_id=vpc.id,
    description="Database cluster security group",
    ingress=[
        # PostgreSQL from vpc
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=5432,
            to_port=5432,
            cidr_blocks=['10.0.0.0/16'],
        ),
        # SSH access from anywhere
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=22,
            to_port=22,
            cidr_blocks=['0.0.0.0/0'],
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
        'Name': 'db-cluster-security-group'
    }
)

db_user_data = """#!/bin/bash
# Update system packages
sudo apt update

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
"""

# Create EC2 Instances for MasterDB
master_instances = []
for i in range(1):
    master = aws.ec2.Instance(
        f'master-{i}',
        instance_type='t2.micro', # Update with correct instance type
        ami='ami-01811d4912b4ccb26',  # Update with correct Ubuntu AMI ID
        subnet_id=subnet.id,
        key_name="db-cluster",
        vpc_security_group_ids=[security_group.id],
        associate_public_ip_address=True,
        private_ip=f'10.0.1.1{i}',
        user_data=db_user_data,
        tags={
            'Name': f'master-{i}'
        }
    )
    master_instances.append(master)

# Create EC2 Instances for Replicas
replica_instances = []
for i in range(2):
    replica = aws.ec2.Instance(
        f'replica-{i}',
        instance_type='t2.micro', # Update with correct instance type
        ami='ami-01811d4912b4ccb26',  # Update with correct Ubuntu AMI ID
        subnet_id=subnet.id,
        key_name="db-cluster",
        vpc_security_group_ids=[security_group.id],
        associate_public_ip_address=True,
        private_ip=f'10.0.1.2{i}',
        user_data=db_user_data,
        tags={'Name': f'replica-{i}'}
    )
    replica_instances.append(replica)

# Export Public and Private IPs of Controller and Worker Instances
master_public_ips = [master.public_ip for master in master_instances]
master_private_ips = [master.private_ip for master in master_instances]
replica_public_ips = [replica.public_ip for replica in replica_instances]
replica_private_ips = [replica.private_ip for replica in replica_instances]

pulumi.export('master_public_ips', master_public_ips)
pulumi.export('master_private_ips', master_private_ips)
pulumi.export('replica_public_ips', replica_public_ips)
pulumi.export('replica_private_ips', replica_private_ips)

# Export the VPC ID and Subnet ID for reference
pulumi.export('vpc_id', vpc.id)
pulumi.export('subnet_id', subnet.id)

# create config file
def create_config_file(args):
    
    # Split the flattened list into IPs and hostnames
    ip_list = args[:len(args)//2]
    hostname_list = args[len(args)//2:]
    
    config_content = ""
    
    # Iterate over IP addresses and corresponding hostnames
    for hostname, ip in zip(hostname_list, ip_list):
        config_content += f"Host {hostname}\n"
        config_content += f"    HostName {ip}\n"
        config_content += f"    User ubuntu\n"
        config_content += f"    IdentityFile ~/.ssh/db-cluster.id_rsa\n\n"
    
    # Write the content to the SSH config file
    config_path = os.path.expanduser("~/.ssh/config")
    with open(config_path, "w") as config_file:
        config_file.write(config_content)

# Collect the IPs for all nodes
all_ips = [master.public_ip for master in master_instances] + [replica.public_ip for replica in replica_instances]

all_hostnames = [master.tags["Name"] for master in master_instances] + \
                [replica.tags["Name"] for replica in replica_instances]

combined_output = all_ips + all_hostnames

# Create the config file with the IPs once the instances are ready
pulumi.Output.all(*combined_output).apply(create_config_file)