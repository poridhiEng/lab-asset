import pulumi
import pulumi_aws as aws
import os

# variables
instance_type = 't2.micro'
ami = 'ami-01811d4912b4ccb26'  # Ubuntu 22.04 LTS in ap-southeast-1
key_name = "db-cluster"

# Create a VPC
vpc = aws.ec2.Vpc(
    'url-shortener-vpc',
    cidr_block='10.0.0.0/16',
    enable_dns_support=True,
    enable_dns_hostnames=True,
    tags={'Name': 'url-shortener-vpc'}
)

# Create public and private subnets
public_subnet = aws.ec2.Subnet(
    'url-shortener-public-subnet',
    vpc_id=vpc.id,
    cidr_block='10.0.1.0/24',
    map_public_ip_on_launch=True,
    availability_zone='ap-southeast-1a',  
    tags={'Name': 'url-shortener-public-subnet'}
)

private_subnet = aws.ec2.Subnet(
    'url-shortener-private-subnet',
    vpc_id=vpc.id,
    cidr_block='10.0.2.0/24',
    map_public_ip_on_launch=False,
    availability_zone='ap-southeast-1a',  
    tags={'Name': 'url-shortener-private-subnet'}
)

# Create an Internet Gateway
internet_gateway = aws.ec2.InternetGateway(
    'url-shortener-internet-gateway',
    vpc_id=vpc.id,
    tags={'Name': 'url-shortener-internet-gateway'}
)

# Create NAT Gateway for private subnet
elastic_ip = aws.ec2.Eip('nat-eip')

nat_gateway = aws.ec2.NatGateway(
    'url-shortener-nat-gateway',
    allocation_id=elastic_ip.id,
    subnet_id=public_subnet.id,
    tags={'Name': 'url-shortener-nat-gateway'},
    opts=pulumi.ResourceOptions(depends_on=[internet_gateway])
)

# Create public Route Table
public_route_table = aws.ec2.RouteTable(
    'url-shortener-public-route-table',
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block='0.0.0.0/0',
            gateway_id=internet_gateway.id,
        )
    ],
    tags={'Name': 'url-shortener-public-route-table'}
)

# Create private Route Table
private_route_table = aws.ec2.RouteTable(
    'url-shortener-private-route-table',
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block='0.0.0.0/0',
            nat_gateway_id=nat_gateway.id,
        )
    ],
    tags={'Name': 'url-shortener-private-route-table'}
)

# Associate route tables with subnets
public_route_table_association = aws.ec2.RouteTableAssociation(
    'url-shortener-public-route-table-association',
    subnet_id=public_subnet.id,
    route_table_id=public_route_table.id
)

private_route_table_association = aws.ec2.RouteTableAssociation(
    'url-shortener-private-route-table-association',
    subnet_id=private_subnet.id,
    route_table_id=private_route_table.id
)

# Create security group for Node.js application
app_security_group = aws.ec2.SecurityGroup(
    'app-security-group',
    vpc_id=vpc.id,
    description="Security group for Node.js application",
    ingress=[
        # SSH access
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=22,
            to_port=22,
            cidr_blocks=['0.0.0.0/0'],  # Consider restricting to your IP
        ),
        # HTTP access
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=80,
            to_port=80,
            cidr_blocks=['0.0.0.0/0'],
        ),
        # HTTPS access
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=443,
            to_port=443,
            cidr_blocks=['0.0.0.0/0'],
        ),
        # Node.js application port (if running directly)
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=3000,
            to_port=3000,
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
    tags={'Name': 'app-security-group'}
)

# Create security group for PostgreSQL database
postgres_db_security_group = aws.ec2.SecurityGroup(
    'postgres-db-security-group',
    vpc_id=vpc.id,
    description="Security group for PostgreSQL database",
    ingress=[
        # aws.ec2.SecurityGroupIngressArgs(
        #     protocol='tcp',
        #     from_port=22,
        #     to_port=22,
        #     cidr_blocks=[public_subnet.cidr_block],
        # ),
        # PostgreSQL access from app server
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=5432,
            to_port=5432,
            security_groups=[app_security_group.id],
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

# Create security group for MongoDB
mongo_db_security_group = aws.ec2.SecurityGroup(
    'mongo-db-security-group',
    vpc_id=vpc.id,
    description="Security group for MongoDB database",
    ingress=[
        # aws.ec2.SecurityGroupIngressArgs(
        #     protocol='tcp',
        #     from_port=22,
        #     to_port=22,
        #     cidr_blocks=[public_subnet.cidr_block],
        # ),
        # MongoDB access from app server
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=27017,
            to_port=27017,
            security_groups=[app_security_group.id],
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
    tags={'Name': 'mongo-db-security-group'}
)

# Create security group for Redis
redis_db_security_group = aws.ec2.SecurityGroup(
    'redis-db-security-group',
    vpc_id=vpc.id,
    description="Security group for Redis database",
    ingress=[
        # aws.ec2.SecurityGroupIngressArgs(
        #     protocol='tcp',
        #     from_port=22,
        #     to_port=22,
        #     cidr_blocks=[public_subnet.cidr_block],
        # ),
        # Redis access from app server
        aws.ec2.SecurityGroupIngressArgs(
            protocol='tcp',
            from_port=6379,
            to_port=6379,
            security_groups=[app_security_group.id],
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
    tags={'Name': 'redis-db-security-group'}
)


# Create db server for postgres and mongodb
mongo_postgres_db = aws.ec2.Instance(
    'mongo-postgres-db-server',
    instance_type=instance_type,
    ami=ami,
    subnet_id=private_subnet.id,
    key_name=key_name,
    vpc_security_group_ids=[
        postgres_db_security_group.id,
        mongo_db_security_group.id
    ],
    user_data="""#!/bin/bash
    # Update system and install Docker
    sudo apt-get update -y
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker

    # Create Docker network for communication
    sudo docker network create url-shortener-net

    # Run MongoDB container
    sudo docker run -d \
        --name url_shortener_mongo \
        --network url-shortener-net \
        -e MONGO_INITDB_DATABASE=url_shortener \
        -v mongo-data:/data/db \
        -p 27017:27017 \
        --restart unless-stopped \
        mongo:6.0

    # Run PostgreSQL container
    sudo docker run -d \
        --name url_shortener_postgres \
        --network url-shortener-net \
        -e POSTGRES_USER=your_pg_user \
        -e POSTGRES_PASSWORD=your_pg_password \
        -e POSTGRES_DB=url_shortener \
        -v postgres-data:/var/lib/postgresql/data \
        -p 5432:5432 \
        --restart unless-stopped \
        postgres:15
    """,
    user_data_replace_on_change=True,
    tags={'Name': 'mongo-postgres-db-server'},
    opts=pulumi.ResourceOptions(
        depends_on=[
            nat_gateway,
            private_route_table_association,
        ]
    )
)

# Create redis server
redis_db = aws.ec2.Instance(
    'redis-db-server',
    instance_type=instance_type,
    ami=ami,
    subnet_id=private_subnet.id,
    key_name=key_name,
    vpc_security_group_ids=[redis_db_security_group.id],
    user_data="""#!/bin/bash
    # Update system and install Docker
    sudo apt-get update -y
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker

    # Run Redis container
    sudo docker run -d \
        --name url_shortener_redis \
        -v redis-data:/data \
        -p 6379:6379 \
        --restart unless-stopped \
        redis:7.0 \
        redis-server --requirepass your_redis_password
    """,
    user_data_replace_on_change=True,
    tags={'Name': 'redis-db-server'},
    opts=pulumi.ResourceOptions(
        depends_on=[
            nat_gateway,
            private_route_table_association,
        ]
    )
)

# Create app server to run nodejs application
app = aws.ec2.Instance(
    'app-server',
    instance_type=instance_type,
    ami=ami,
    subnet_id=public_subnet.id,
    key_name=key_name,
    vpc_security_group_ids=[app_security_group.id],
    associate_public_ip_address=True,
    user_data="""#!/bin/bash
    sudo apt update
    sudo apt install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
    """,
    user_data_replace_on_change=True,
    tags={'Name': 'app-server'}
)

# Export Public and Private IPs
pulumi.export('app_public_ip', app.public_ip)
pulumi.export('app_private_ip', app.private_ip)
pulumi.export('mongo_postgres_db_private_ip', mongo_postgres_db.private_ip)
pulumi.export('redis_db_private_ip', redis_db.private_ip)

# Export the VPC ID and Subnet IDs for reference
pulumi.export('vpc_id', vpc.id)
pulumi.export('public_subnet_id', public_subnet.id)
pulumi.export('private_subnet_id', private_subnet.id)

# Create config file for SSH access
def create_config_file(ips):
    config_content = f"""Host app-server
    HostName {ips['app_public_ip']}
    User ubuntu
    IdentityFile ~/.ssh/{key_name}.id_rsa

Host mongo-postgres-db-server
    ProxyJump app-server
    HostName {ips['mongo_postgres_db_private_ip']}
    User ubuntu
    IdentityFile ~/.ssh/{key_name}.id_rsa

Host redis-db-server
    ProxyJump app-server
    HostName {ips['redis_db_private_ip']}
    User ubuntu
    IdentityFile ~/.ssh/{key_name}.id_rsa
"""
    
    config_path = os.path.expanduser("~/.ssh/config")
    with open(config_path, "w") as config_file:
        config_file.write(config_content)

# Collect the IPs for all nodes
all_ips = pulumi.Output.all(
    app_public_ip=app.public_ip,
    mongo_postgres_db_private_ip=mongo_postgres_db.private_ip,
    redis_db_private_ip=redis_db.private_ip
)

# Create the config file with the IPs once the instances are ready
all_ips.apply(create_config_file)