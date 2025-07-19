import pulumi
import pulumi_aws as aws

# Create Workload VPC
workload_vpc = aws.ec2.Vpc("workload_vpc", 
    cidr_block="10.1.0.0/16",
    enable_dns_support=True,
    enable_dns_hostnames=True, 
    tags={"Name": "WorkloadVPC"}
)

# Create Egress VPC
egress_vpc = aws.ec2.Vpc("egress_vpc", 
    cidr_block="10.2.0.0/16",
    enable_dns_support=True,
    enable_dns_hostnames=True,
    tags={"Name": "EgressVPC"}
)

# Create Subnets in Workload VPC
workload_private_subnet = aws.ec2.Subnet("workload_private_subnet",
    vpc_id=workload_vpc.id,
    cidr_block="10.1.1.0/24",
    availability_zone="ap-southeast-1a",
    tags={"Name": "WorkloadPrivateSubnet"}
)

workload_transit_subnet = aws.ec2.Subnet("workload_transit_subnet",
    vpc_id=workload_vpc.id,
    cidr_block="10.1.2.0/24",
    availability_zone="ap-southeast-1b",
    tags={"Name": "WorkloadTransitSubnet"}

)

# Create Subnets in Egress VPC
egress_transit_subnet = aws.ec2.Subnet("egress_transit_subnet",
    vpc_id=egress_vpc.id,
    cidr_block="10.2.1.0/24",
    availability_zone="ap-southeast-1a",
    tags={"Name": "EgressTransitSubnet"}
)

egress_NAT_subnet = aws.ec2.Subnet("egress_NAT_subnet",
    vpc_id=egress_vpc.id,
    cidr_block="10.2.2.0/24",
    availability_zone="ap-southeast-1a",
    map_public_ip_on_launch=True,
    tags={"Name": "EgressNATSubnet"}
)

# Create Internet Gateway
igw = aws.ec2.InternetGateway("egress_igw", vpc_id=egress_vpc.id,  tags={"Name": "EgressIGW"})

# Create NAT Gateway
nat_eip = aws.ec2.Eip("nat_eip")
nat_gateway = aws.ec2.NatGateway("egress_nat_gateway",
    allocation_id=nat_eip.id,
    subnet_id=egress_NAT_subnet.id,
    tags={"Name": "EgressNATGW"}
)

# Create Transit Gateway
tgw = aws.ec2transitgateway.TransitGateway("aws_tgw", tags={"Name": "WorkloadEgressTGW"})

# Attachments to Transit Gateway
workload_tgw_attachment = aws.ec2transitgateway.VpcAttachment("workload_attachment",
    transit_gateway_id=tgw.id,
    vpc_id=workload_vpc.id,
    subnet_ids=[workload_transit_subnet.id],
    tags={"Name": "WorkloadTGWAttachment"}
)

egress_tgw_attachment = aws.ec2transitgateway.VpcAttachment("egress_attachment",
    transit_gateway_id=tgw.id,
    vpc_id=egress_vpc.id,
    subnet_ids=[egress_transit_subnet.id],
    tags={"Name": "EgresssTGWAttachment"}
)

# Route table for Workload Private Subnet 1
workload_private_rt = aws.ec2.RouteTable("workload_private_rt",
    vpc_id=workload_vpc.id,
    routes=[{"cidr_block": "0.0.0.0/0", "transit_gateway_id": tgw.id}],
    tags={"Name": "workloadPrivateRouteTable"}
)

# Route table association for Workload Private Subnet 1
aws.ec2.RouteTableAssociation("workload_private_rt_assoc",
    subnet_id=workload_private_subnet.id,
    route_table_id=workload_private_rt.id
)

# Route table for Workload Private Subnet 2
workload_transit_rt = aws.ec2.RouteTable("workload_transit_rt",
    vpc_id=workload_vpc.id,
    # routes=[{"cidr_block": "0.0.0.0/0", "transit_gateway_id": tgw.id}],
    tags={"Name": "workloadTransitRouteTable"}
)

# Route table association for Workload Private Subnet 2
aws.ec2.RouteTableAssociation("workload_transit_rt_assoc",
    subnet_id=workload_private_subnet.id,
    route_table_id=workload_transit_rt.id
)

# Route table for Egress Private Subnet
egress_transit_rt = aws.ec2.RouteTable("egress_transit_rt",
    vpc_id=egress_vpc.id,
    routes=[{"cidr_block": "0.0.0.0/0", "nat_gateway_id": nat_gateway.id}],
    tags={"Name": "EgressTransitRouteTable"}
)

# Route table association for Egress Private Subnet
aws.ec2.RouteTableAssociation("egress_transit_rt_assoc",
    subnet_id=egress_transit_subnet.id,
    route_table_id=egress_transit_rt.id
)

# Route table for Egress Public Subnet
egress_nat_rt = aws.ec2.RouteTable("egress_nat_rt",
    vpc_id=egress_vpc.id,
    routes=[
        {"cidr_block": "0.0.0.0/0", "gateway_id": igw.id},
        {"cidr_block": "10.1.0.0/16", "transit_gateway_id": tgw.id}
    ],

    tags={"Name": "EgressNATRouteTable"}
)

# Route table association for Egress Public Subnet
aws.ec2.RouteTableAssociation("egress_nat_rt_assoc",
    subnet_id=egress_NAT_subnet.id,
    route_table_id=egress_nat_rt.id
)


security_group1 = aws.ec2.SecurityGroup("my-sec-group1",
    vpc_id=workload_vpc.id,
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(protocol="tcp", from_port=0, to_port=65535, cidr_blocks=["0.0.0.0/0"]),
        aws.ec2.SecurityGroupIngressArgs(protocol="icmp", from_port=-1, to_port=-1, cidr_blocks=["0.0.0.0/0"]),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(protocol="-1", from_port=0, to_port=0, cidr_blocks=["0.0.0.0/0"])
    ]
)

# EC2 Instance in Workload VPC
workload_instance = aws.ec2.Instance("workload_instance",
    instance_type="t2.micro",
    ami="ami-0672fd5b9210aa093",
    vpc_security_group_ids=[security_group1.id],
    subnet_id=workload_private_subnet.id,
    key_name="key-pair-poridhi-poc", 
    associate_public_ip_address=False,
    tags={"Name": "WorkloadInstance"}
)

pulumi.export("workload_vpc_id", workload_vpc.id)
pulumi.export("egress_vpc_id", egress_vpc.id)
pulumi.export("transit_gateway_id", tgw.id)



