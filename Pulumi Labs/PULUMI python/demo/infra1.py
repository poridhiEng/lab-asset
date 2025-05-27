import pulumi
import pulumi_aws as aws

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

public_subnet1 = aws.ec2.Subnet(
    'public-subnet-1',
    vpc_id=vpc.id,
    cidr_block='10.0.1.0/24',
    availability_zone='ap-southeast-1a',
    map_public_ip_on_launch=True,
    tags={'Name': 'public-subnet-1'}
)

public_subnet2 = aws.ec2.Subnet(
    'public-subnet-2',
    vpc_id=vpc.id,
    cidr_block='10.0.2.0/24',
    availability_zone='ap-southeast-1b',
    map_public_ip_on_launch=True,
    tags={'Name': 'public-subnet-2'}
)

# Create an Internet Gateway
internet_gateway = aws.ec2.InternetGateway(
    'cluster-internet-gateway',
    vpc_id=vpc.id,
    tags={'Name': 'cluster-internet-gateway'}
)

# Create a Route Table with a default route to the Internet Gateway
public_route_table = aws.ec2.RouteTable(
    'public-route-table',
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block='0.0.0.0/0',
            gateway_id=internet_gateway.id,
        )
    ],
    tags={'Name': 'public-route-table'}
)

# Associate both subnets with the public route table
route_table_association1 = aws.ec2.RouteTableAssociation(
    'public-subnet-1-route-association',
    subnet_id=public_subnet1.id,
    route_table_id=public_route_table.id
)

route_table_association2 = aws.ec2.RouteTableAssociation(
    'public-subnet-2-route-association',
    subnet_id=public_subnet2.id,
    route_table_id=public_route_table.id
)

# Create a security group allowing:
security_group = aws.ec2.SecurityGroup(
    'cluster-security-group',
    vpc_id=vpc.id,
    description="Cluster security group",
    ingress=[
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
    tags={'Name': 'cluster-security-group'}
)

# Create two EC2 instances in different subnets
instance1 = aws.ec2.Instance(
    'instance-1',
    instance_type=instance_type,
    ami=ami,
    subnet_id=public_subnet1.id,
    key_name=key_name,
    vpc_security_group_ids=[security_group.id],
    associate_public_ip_address=True,
    private_ip='10.0.1.10',  # Private IP in subnet 1
    tags={'Name': 'instance-1'}
)

instance2 = aws.ec2.Instance(
    'instance-2',
    instance_type=instance_type,
    ami=ami,
    subnet_id=public_subnet2.id,
    key_name=key_name,
    vpc_security_group_ids=[security_group.id],
    associate_public_ip_address=True,
    private_ip='10.0.2.10',  # Private IP in subnet 2
    tags={'Name': 'instance-2'}
)

# Export instance details
pulumi.export('instance1_public_ip', instance1.public_ip)
pulumi.export('instance2_public_ip', instance2.public_ip)
pulumi.export('instance1_private_ip', instance1.private_ip)
pulumi.export('instance2_private_ip', instance2.private_ip)