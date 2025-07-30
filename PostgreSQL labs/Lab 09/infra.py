import pulumi
import pulumi_aws as aws
import os

# Variables
instance_type = 't2.micro'
ami = 'ami-01811d4912b4ccb26'  # Replace with the correct AMI for your region

# Create VPC
vpc = aws.ec2.Vpc(
    'postgres-db-vpc',
    cidr_block='10.0.0.0/16',
    enable_dns_support=True,
    enable_dns_hostnames=True,
    tags={'Name': 'postgres-db-vpc'}
)

# Create public and private subnets
public_subnet = aws.ec2.Subnet(
    'postgres-db-public-subnet',
    vpc_id=vpc.id,
    cidr_block='10.0.1.0/24',
    map_public_ip_on_launch=True,
    availability_zone='ap-southeast-1a',  
    tags={'Name': 'postgres-db-public-subnet'}
)

private_subnet = aws.ec2.Subnet(
    'postgres-db-private-subnet',
    vpc_id=vpc.id,
    cidr_block='10.0.2.0/24',
    map_public_ip_on_launch=False,
    availability_zone='ap-southeast-1a',  
    tags={'Name': 'postgres-db-private-subnet'}
)

# Create an Internet Gateway
internet_gateway = aws.ec2.InternetGateway(
    'postgres-db-internet-gateway',
    vpc_id=vpc.id,
    tags={'Name': 'postgres-db-internet-gateway'}
)

# Create NAT Gateway for private subnet
elastic_ip = aws.ec2.Eip('nat-eip')

nat_gateway = aws.ec2.NatGateway(
    'postgres-db-nat-gateway',
    allocation_id=elastic_ip.id,
    subnet_id=public_subnet.id,
    tags={'Name': 'postgres-db-nat-gateway'}
)

# Create public Route Table
public_route_table = aws.ec2.RouteTable(
    'postgres-db-public-route-table',
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block='0.0.0.0/0',
            gateway_id=internet_gateway.id,
        )
    ],
    tags={'Name': 'postgres-db-public-route-table'}
)

# Create private Route Table
private_route_table = aws.ec2.RouteTable(
    'postgres-db-private-route-table',
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block='0.0.0.0/0',
            nat_gateway_id=nat_gateway.id,
        )
    ],
    tags={'Name': 'postgres-db-private-route-table'}
)

# Associate route tables with subnets
public_route_table_association = aws.ec2.RouteTableAssociation(
    'postgres-db-public-route-table-association',
    subnet_id=public_subnet.id,
    route_table_id=public_route_table.id
)

private_route_table_association = aws.ec2.RouteTableAssociation(
    'postgres-db-private-route-table-association',
    subnet_id=private_subnet.id,
    route_table_id=private_route_table.id
)

# Create security group for PostgreSQL database
postgres_db_security_group = aws.ec2.SecurityGroup(
    'postgres-db-security-group',
    vpc_id=vpc.id,
    description="Security group for PostgreSQL database",
    ingress=[
        # SSH access from monitoring server
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=22,
            to_port=22,
            cidr_blocks=[public_subnet.cidr_block],
        ),
        # PostgreSQL access from monitoring server
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=5432,
            to_port=5432,
            cidr_blocks=[public_subnet.cidr_block],
        ),
        # PgBouncer access from monitoring server
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=6432,
            to_port=6432,
            cidr_blocks=[public_subnet.cidr_block],
        ),
        # PostgreSQL Exporter access from monitoring server
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=9187,
            to_port=9187,
            cidr_blocks=[public_subnet.cidr_block],
        ),
        # PgBouncer Exporter access from monitoring server
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=9127,
            to_port=9127,
            cidr_blocks=[public_subnet.cidr_block],
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
    tags={'Name': 'postgres-db-security-group'}
)

# Create security group for monitoring server
monitoring_security_group = aws.ec2.SecurityGroup(
    'monitoring-security-group',
    vpc_id=vpc.id,
    description="Security group for monitoring server",
    ingress=[
        # SSH access
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=22,
            to_port=22,
            cidr_blocks=['0.0.0.0/0'],
        ),
        # Grafana access
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=3000,
            to_port=3000,
            cidr_blocks=['0.0.0.0/0'],      
        ),
        # Prometheus access
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=9090,
            to_port=9090,
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
    tags={'Name': 'monitoring-security-group'}
)

# Create PostgreSQL + PgBouncer + Exporters server
db = aws.ec2.Instance(
    'postgres-db-server',
    instance_type=instance_type,
    ami=ami,
    subnet_id=private_subnet.id,
    key_name="db-cluster",
    vpc_security_group_ids=[postgres_db_security_group.id],
    user_data_replace_on_change=True,
    user_data="""#!/bin/bash
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib pgbouncer
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    sudo systemctl start pgbouncer
    sudo systemctl enable pgbouncer
    """,
    tags={'Name': 'postgres-db-server'},
    opts=pulumi.ResourceOptions(
        depends_on=[
            nat_gateway,
            private_route_table_association,
            private_subnet
        ]
    )
)

# Create monitoring server
monitoring_server = aws.ec2.Instance(
    'monitoring-server',
    instance_type=instance_type,
    ami=ami,
    subnet_id=public_subnet.id,
    key_name="db-cluster",
    vpc_security_group_ids=[monitoring_security_group.id],
    associate_public_ip_address=True,
    user_data_replace_on_change=True,
    user_data="""#!/bin/bash
    sudo apt update
    sudo apt install -y docker.io docker-compose
    sudo systemctl enable docker
    sudo systemctl start docker
    """,
    tags={'Name': 'monitoring-server'}
)

# Export Public and Private IPs
pulumi.export('monitoring_server_public_ip', monitoring_server.public_ip)
pulumi.export('monitoring_server_private_ip', monitoring_server.private_ip)
pulumi.export('postgres_db_private_ip', db.private_ip)

# Export the VPC ID and Subnet IDs for reference
pulumi.export('vpc_id', vpc.id)
pulumi.export('public_subnet_id', public_subnet.id)
pulumi.export('private_subnet_id', private_subnet.id)

# Create config file
def create_config_file(all_ips):
    config_content = f"""Host monitoring-server
    HostName {all_ips[0]}
    User ubuntu
    IdentityFile ~/.ssh/db-cluster.id_rsa

Host db-server
    ProxyJump monitoring-server
    HostName {all_ips[1]}
    User ubuntu
    IdentityFile ~/.ssh/db-cluster.id_rsa
"""
    
    config_path = os.path.expanduser("~/.ssh/config")
    with open(config_path, "w") as config_file:
        config_file.write(config_content)

# Collect the IPs for all nodes
all_ips = [monitoring_server.public_ip, db.private_ip]

# Create the config file with the IPs once the instances are ready
pulumi.Output.all(*all_ips).apply(create_config_file)