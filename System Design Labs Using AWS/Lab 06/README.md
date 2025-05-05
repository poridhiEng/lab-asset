# **PgBouncer: Enhancing PostgreSQL Performance**

If you're working with large-scale applications, managing concurrent database connections efficiently is crucial. PostgreSQL performs well on its own, handling up to 350 transactions per second, but when the demand exceeds this limit, the number of processes grows exponentially, impacting system performance.  

This is where **PgBouncer** comes in. As a lightweight connection pooler, PgBouncer helps optimize database connections, reducing resource consumption and improving performance.

![](./images/db-conn3.drawio.svg)

In this lab, we’ll cover the fundamental concepts of PgBouncer, how to install and configure it, and explore advanced settings for optimal performance and benchmark testing.

## **Why Use PgBouncer?**

The core idea behind PgBouncer is **connection pooling**—it helps manage database connections efficiently by keeping an array of open connections and reusing them, instead of creating and destroying them for each request.

### **The Problem with Direct PostgreSQL Connections**

Each time a PostgreSQL client initiates a new database connection, the system performs the following tasks:  
- Allocates memory (a few hundred KB per connection).  
- Executes a **fork** system call, creating a process with a new memory space.  
- Assigns a file descriptor and memory segment.  
- Terminates the process when the connection is closed.  

While this process is efficient for a small number of connections, large-scale applications often make short-lived queries, which means opening and closing thousands of connections per second. This can significantly slow down performance.

![](./images/db-conn.drawio.svg)



### **How PgBouncer Solves This Problem**

PgBouncer acts as a **proxy** between applications and the PostgreSQL database:

- It maintains a pool of persistent database connections.  
- When an application requests a connection, PgBouncer assigns an already-open connection from the pool.  
- When the application finishes using the connection, it is returned to the pool instead of being closed.  
- This approach drastically reduces the overhead of creating and closing connections.  

Additionally, PgBouncer has a very low memory footprint—some users report as low as **2 KB per connection**, making it an ideal solution for high-throughput applications.

## Hands-on Implementation of PgBouncer with AWS Infrastructure

In this implementation, we will set up a robust AWS infrastructure designed for optimizing PostgreSQL database connections using **PgBouncer** as a connection pooling solution. We will integrate **Prometheus** for monitoring and **Grafana** for visualizing metrics. This solution will include multiple components to ensure scalability, performance, and ease of monitoring. Below is an overview of the components involved:

### Components:

- **PostgreSQL Database**: The core relational database to store application data.
  
- **PgBouncer (Connection Pooling)**: A lightweight connection pooler for PostgreSQL to manage database connections efficiently, especially for applications that make frequent database requests.
  
- **Prometheus (Metrics Collection)**: A monitoring and alerting toolkit designed to collect and store metrics over time. It will scrape and store metrics from both PostgreSQL and PgBouncer.
  
- **Grafana (Visualization)**: A data visualization platform that will query Prometheus to display database and connection metrics in a graphical format, enabling real-time monitoring.
  
- **PostgreSQL Exporter (Metrics for PostgreSQL)**: An exporter that exposes PostgreSQL-specific metrics, such as database performance, query execution times, and server health.
  
- **PgBouncer Exporter (Metrics for PgBouncer)**: An exporter that collects and exposes metrics related to PgBouncer, such as connection pool statistics and query performance.

## AWS Infrastructure Architecture

The infrastructure setup will utilize **two EC2 instances** within an AWS Virtual Private Cloud (VPC). Instance 1 will host the **PostgreSQL database**, **PgBouncer**, and the **exporters** for Prometheus. Instance 2 will host **Prometheus** (for scraping and storing metrics) and **Grafana** (for visualization). The instances will be placed in different subnets to ensure proper isolation and security.

![](./images/arch.drawio.svg)

### Basic Setup Overview:

- **Instance 1**: PostgreSQL + PgBouncer + Exporters  
  This instance will reside in a **private subnet** and will run the PostgreSQL database, PgBouncer, and the necessary exporters that provide metrics to Prometheus. As it resides in the private subnet, it will not have direct internet access, and a **NAT Gateway** will be used to allow internet connectivity for software updates, etc.

  To allow communication between the different components securely, we need to configure the security group for Instance 1 to allow inbound traffic on the following ports:
  - **Port 5432**: The default port for PostgreSQL communication between the database and client applications.
  - **Port 6432**: The default port for PgBouncer, allowing client applications to connect through the pooler instead of directly to the PostgreSQL database.
  - **Port 9187**: The port used by the **PostgreSQL Exporter** to expose PostgreSQL metrics to Prometheus.
  - **Port 9127**: The port used by the **PgBouncer Exporter** to expose PgBouncer metrics to Prometheus.

- **Instance 2**: Prometheus + Grafana  
  This instance will be located in the **public subnet**, making it accessible for incoming traffic. It will run Prometheus to scrape metrics from the exporters and Grafana for visualizing these metrics. This instance will have internet access, allowing it to retrieve updates and interact with external systems if needed.

  To enable secure communication while restricting access to trusted sources, the security group for Instance 2 will be configured as follows:
  - **Port 9090**: This is the default port for Prometheus, allowing users to access the Prometheus dashboard to query and analyze metrics.
  - **Port 3000**: This is the default port for Grafana, providing access to the Grafana dashboard for visualizing the metrics collected by Prometheus.

## Environment Setup for Infrastructure Provisioning

### Configure AWS CLI

The AWS CLI is a command-line tool that allows you to interact with AWS services programmatically. It simplifies provisioning resources, such as EC2 instances and load balancers, which are required to host our database cluster and application server. Lets configure the AWS CLI.

```bash
aws configure
```

- `AWS Access Key ID:` Your access key to authenticate AWS API requests.
- `AWS Secret Access Key:` A secret key associated with your access key.
- `Default region:` The AWS region in which you want to provision your resources (ap-southeast-1).
- `Default output format:` You can choose the output format (JSON, text, table).

> Get your access key and secret access key from `Poridhi's lab` account by generating AWS credentials.

![alt text](./images/image-4.png)


### Provisioning Compute Resources

**1 Create a Directory for Your Infrastructure**

Before starting, it’s best to create a dedicated directory for the infrastructure files:

```bash
mkdir db-infra
cd db-infra
```

**2. Install Python venv**

Set up a Python virtual environment (venv) to manage dependencies for Pulumi or other Python-based tools:

```bash
sudo apt update
sudo apt install python3.8-venv -y
```

This will set up a Python virtual environment which will be useful later when working with Pulumi.

**3. Initialize a New Pulumi Project**

First login to Pulumi by running the command in the terminal. You will require a **token** to login. You can get the token from the Pulumi website.

```bash
pulumi login
```

![alt text](./images/image-5.png)

After login, run the following command to initialize a new Pulumi project:

```bash
pulumi new aws-python
```

You will prompted with serveral questions. Go with the default options. Only change the aws region to `ap-southeast-1`.

![alt text](./images/image-6.png)

**4. Update the `__main.py__` file:**

Open the `__main__.py` file in the project directory and define the AWS infrastructure required for our project. It handles the creation of a Virtual Private Cloud (VPC), subnets, security group, NAT Gateway, Internet Gateway, Route Table, and EC2 instances. We will also inject user data to each instances that will install some required tools like postgres database, pgbouncer, docker.

```python
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

Host postgres-db-server
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
```

This pulumi code will create a config file in your `~/.ssh/config` file with the IPs of the DB and Moniroting servers. It will automatically add the `IPs` to the config file when the instances are ready.


**5. Create an AWS Key Pair**

Database Cluster nodes need to communicate securely. This key pair will be used to authenticate when accessing EC2 instances. Use the following AWS CLI command to create a new SSH key pair named kubernetes:

```bash
cd ~/.ssh/
aws ec2 create-key-pair --key-name db-cluster --output text --query 'KeyMaterial' > db-cluster.id_rsa
chmod 400 db-cluster.id_rsa
```

![alt text](./images/image-7.png)

This will save the private key as `db-cluster.id_rsa` in the `~/.ssh/` directory and restrict its permissions.


**6. Provision the Infrastructure**

Run the following command to provision the infrastructure:

```bash
pulumi up --yes
```

![alt text](./images/image-9.png)

Here we can see the infrastructure is provisioned successfully. Now we can use the config file to access the pgadmin server.

#### Check the config file

After the infrastructure is provisioned, you can check the config file to see the IPs of the pgadmin server.

```bash
cat ~/.ssh/config
```

![alt text](./images/image-8.png)


**7. SSH into the ec2 instance**

We can use the config file to access the pgadmin server.

```bash
ssh monitoring-server
```

SSH into the db server.

```bash
ssh db-server
```


You may change the hostname to `monitoring-server` and `db-server` to make it easier to identify the server. This is optional.

```bash
sudo hostnamectl set-hostname monitoring-server
sudo hostnamectl set-hostname db-server
```

![alt text](./images/image-10.png)


## **Set Up PostgreSQL + PgBouncer + Exporters Server**

The goal of this step is to configure and deploy PostgreSQL, PgBouncer (a connection pooler), and exporters that will collect metrics for monitoring with Prometheus.

### **SSH into the PostgreSQL Instance**

Before starting the configuration, SSH into the db instance

```bash
ssh db-server
```

Check the cloud-init log for successfull user data execution.

### **Configure PostgreSQL**

#### **1. Edit PostgreSQL Configuration**

PostgreSQL needs to be configured to accept connections from external sources, including the PgBouncer instance and monitoring servers. This is done by editing the main PostgreSQL configuration file (`postgresql.conf`).

**1.1 Open the PostgreSQL configuration file with the following command:**

```bash
sudo vim /etc/postgresql/*/main/postgresql.conf
```
**1.2. Locate and modify the following settings:**

- `listen_addresses`: This setting defines which IP addresses PostgreSQL will listen to for incoming connections. You can either allow connections from all IPs or restrict it to specific IP ranges (like your private subnet).
  ```ini
  listen_addresses = '*'  # Accept connections from all IPs (or restrict to private subnet)
  ```
- `max_connections`: This setting controls the maximum number of concurrent connections PostgreSQL can handle. Set it based on the expected load.
  ```ini
  max_connections = 100   # Adjust the value based on your needs
  ```

- `password_encryption`: Since PostgreSQL 10+, passwords default to SCRAM-SHA-256, you must explicitly set the password to MD5. PgBouncer only supports MD5, not SCRAM-SHA-256.

  ```ini
  password_encryption = md5
  ```

**3. After making the changes, save and exit the editor (`:wq` in vim).**

![Configuration Example](./images/image.png)

#### **2. Update `pg_hba.conf`**

Next, configure the `pg_hba.conf` file to allow access from the monitoring server or other trusted hosts.

**2.1. Open the `pg_hba.conf` file:**

```bash
sudo vim /etc/postgresql/*/main/pg_hba.conf
```
**2.2. Add a line that grants access from the monitoring server.** 

Update the IP address with your monitoring serverer Private IP.

```plaintext
host    all             all             <MONITORING_SERVER_PRIVATE_IP>/32       md5
```

Also change the password_encryption to **md5**

![alt text](./images/image-17.png)

Save and close the file.

#### **3. Restart PostgreSQL**

After configuring PostgreSQL, restart the service to apply the changes:

```bash
sudo systemctl restart postgresql
```

![alt text](./images/image-12.png)

#### **4. Set Up a Database and User**

Now, create a database and a user in PostgreSQL that will be used by PgBouncer and the monitoring system.

1. Switch to the `postgres` user:
   ```bash
   sudo -u postgres psql
   ```
2. Create a new database and user, and assign privileges:

   ```sql
   CREATE DATABASE mydb;
   CREATE USER myuser WITH ENCRYPTED PASSWORD 'mypassword';
   GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
   ```

   - The database `mydb` will be used by PgBouncer.
   - The user `myuser` will be granted full access to `mydb`.

   ![Database Setup Example](./images/image-1.png)

3. Exit the psql terminal

## **Configure PgBouncer**

#### **1. Edit PgBouncer Configuration**

Now, configure PgBouncer to act as a connection pooler for PostgreSQL. PgBouncer will handle database connections efficiently and reduce the load on PostgreSQL by reusing database connections.

**1.1. Open the PgBouncer configuration file:**
```bash
sudo vim /etc/pgbouncer/pgbouncer.ini
```
**1.2. Modify the file to reflect your environment:**

```ini
[databases]
mydb = host=127.0.0.1 port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0       
listen_port = 6432             
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt  
pool_mode = transaction        
max_client_conn = 100          
default_pool_size = 20         
stats_users = myuser
ignore_startup_parameters = extra_float_digits
```

> Make sure to update the dbname, stats_users according to your preference. Or keep it as it is.

![alt text](./images/image-20.png)

#### **2. Create a Userlist File**

PgBouncer requires a userlist file that contains the usernames and passwords for authentication. This file is referenced in the PgBouncer configuration.

**2.1. Create the `userlist.txt` file in `/etc/pgbouncer`:**

```bash
echo '"myuser" "md554fdba1dac1d18ce9c2d57a03502a8ff"' | sudo tee /etc/pgbouncer/userlist.txt
```

Here, `myuser` is the PostgreSQL user you created earlier, and the MD5 hash represents the password `mypassword`.

**2. To generate the MD5 hash of the password, run:**

```bash
echo -n "mypasswordmyuser" | md5sum | awk '{print "md5" $1}'
```

- Replace `"mypassword"` with the actual password of the PostgreSQL user.

![alt text](./images/image-16.png)

#### **3. Restart PgBouncer**

After updating the configuration, restart the PgBouncer service to apply the changes:

```bash
sudo systemctl restart pgbouncer
```

![alt text](./images/image-15.png)


#### Verify Connectivity

Now connect to Postgresql database through pgbouncer using our created user and password.

![alt text](./images/image-18.png)

> If you face any issue during authentication, make sure to Change PostgreSQL to Use MD5 Authentication, as pgBouncer only supports MD5

## **Install and Configure Exporters**

To monitor PostgreSQL and PgBouncer, you need to install Prometheus exporters that expose the relevant metrics.

#### **Install PostgreSQL Exporter**
The PostgreSQL exporter collects and exposes metrics from the PostgreSQL database, such as query times, cache hits, and connection statistics.

**Download and install the PostgreSQL exporter:**

```bash
wget https://github.com/prometheus-community/postgres_exporter/releases/download/v0.10.1/postgres_exporter-0.10.1.linux-amd64.tar.gz
tar xvfz postgres_exporter-*.tar.gz
sudo mv postgres_exporter-*/postgres_exporter /usr/local/bin/
sudo chmod +x /usr/local/bin/postgres_exporter
```

![alt text](./images/image-19.png)

> If you face any issue installing exporters, make sure to use the updated version. Visit the github repository to get the latest version.

#### **Install PgBouncer Exporter**

The PgBouncer exporter exposes metrics related to connection pooling and the number of active connections in the pool.

**Download and install the PgBouncer exporter:**

```bash
wget https://github.com/prometheus-community/pgbouncer_exporter/releases/download/v0.10.2/pgbouncer_exporter-0.10.2.linux-amd64.tar.gz
tar xvfz pgbouncer_exporter-0.10.2.linux-amd64.tar.gz
sudo mv pgbouncer_exporter-0.10.2.linux-amd64/pgbouncer_exporter /usr/local/bin/
sudo chmod +x /usr/local/bin/pgbouncer_exporter
```

![Exporter Installation Example](./images/image-3.png)

#### **Start the Exporters**

Start both the PostgreSQL and PgBouncer exporters to begin exposing metrics.

**1. Start the PostgreSQL exporter:**

```bash
export DATA_SOURCE_NAME="postgresql://myuser:mypassword@localhost:5432/mydb?sslmode=disable"
nohup postgres_exporter --web.listen-address=":9187" > /dev/null 2>&1 &
```

**2. Start the PgBouncer exporter:**

```bash
pgbouncer_exporter --pgBouncer.connectionString="postgresql://myuser:mypassword@localhost:6432/pgbouncer?sslmode=disable"
```

The exporters will now run, making metrics available on the specified ports (9187 for PostgreSQL, 9127 for PgBouncer) for Prometheus to scrape.

Check the metrics endpoint:

![alt text](./images/image-21.png)

## **Set Up Monitoring Server**

In this section, we will set up a monitoring server that runs both **Prometheus** (for collecting metrics) and **Grafana** (for visualizing those metrics). We will use **Docker** to manage both Prometheus and Grafana containers. These services will pull metrics from the previously configured PostgreSQL and PgBouncer exporters.

### **Step 1: SSH into the Monitoring Server**

Start by connecting to the server designated for monitoring. This server will host both Prometheus and Grafana.

```bash
ssh monitoring-server
```

This will establish an SSH connection to the monitoring server where we will set up the tools.

### **Step 2: Set Up Prometheus and Grafana**

#### **1. Create a Directory for Configuration**

We will start by creating a directory to hold all of our configuration files for Prometheus and Grafana. This directory will also be used to store Docker volumes.

```bash
mkdir ~/monitoring
cd ~/monitoring
```

This will create a new folder named `monitoring` and navigate into it.

#### **2. Create `docker-compose.yml`**

Next, we will create a `docker-compose.yml` file. This file defines the setup for running both Prometheus and Grafana containers via Docker.

```bash
vim docker-compose.yml
```

Add the following content to the file:

```yaml
version: "3"
services:
  prometheus:
    image: prom/prometheus
    container_name: prometheus
    ports:
      - "9090:9090"   # Expose Prometheus UI on port 9090
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml   # Mount the Prometheus configuration file
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"    # Use the custom configuration

  grafana:
    image: grafana/grafana
    container_name: grafana
    ports:
      - "3000:3000"   # Expose Grafana UI on port 3000
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin  # Set Grafana admin password
```

**Explanation:**

- **Prometheus**: We are using the official `prom/prometheus` Docker image. Prometheus is configured to use the custom configuration file (`prometheus.yml`), which we will create next.
- **Grafana**: The official `grafana/grafana` Docker image is used to run Grafana. We expose port 3000 for the Grafana web UI and set an admin password (`admin`).

#### **3. Create `prometheus.yml`**

Now, create the Prometheus configuration file (`prometheus.yml`). This file tells Prometheus where to scrape metrics from.

```bash
vim prometheus.yml
```

Add the following content to configure Prometheus to scrape metrics from both the **PostgreSQL Exporter** and the **PgBouncer Exporter**:

```yaml
global:
  scrape_interval: 15s  # Scrape metrics every 15 seconds

scrape_configs:
  - job_name: "postgres"
    static_configs:
      - targets: ["10.0.2.58:9187"]  # IP of the PostgreSQL exporter

  - job_name: "pgbouncer"
    static_configs:
      - targets: ["10.0.2.58:9127"]  # IP of the PgBouncer exporter
```

Explanation:
- **global**: Defines a global setting to scrape metrics every 15 seconds.
- **scrape_configs**: This section defines the targets that Prometheus will scrape. In this case, it is scraping metrics from the PostgreSQL and PgBouncer exporters that are exposed on ports `9187` and `9127` respectively. 


> The IP address `10.0.2.58` should be replaced with the db-server private IP where the exporters are running.

#### **4. Start Containers**

After creating both configuration files (`docker-compose.yml` and `prometheus.yml`), we will use **Docker Compose** to launch both Prometheus and Grafana.

```bash
sudo docker-compose up -d
```

This command will download the necessary Docker images (if they are not already present) and start the Prometheus and Grafana containers in the background (`-d` flag stands for "detached mode").

![alt text](./images/image-22.png)

### **Step 3: Verify the Setup**

Now that Prometheus and Grafana are running, let's verify that everything is set up correctly.

#### **1. Access Prometheus**

Open your browser and navigate to Prometheus' web interface. Replace `<monitoring_server_public_ip>` with the actual public IP address of your monitoring server:

```
http://<monitoring_server_public_ip>:9090
```

Prometheus' web UI should load. To check if Prometheus is successfully scraping metrics, go to the **Status > Targets** section in the UI. Here, you should see both the PostgreSQL and PgBouncer exporters listed as "up" if they are being scraped correctly.

![alt text](./images/image-23.png)

#### **2. Access Grafana**

Next, open your browser and navigate to Grafana's web UI:

```
http://<monitoring_server_public_ip>:3000
```

The default login credentials are:
- **Username**: `admin`
- **Password**: `admin`

After logging in, you will be prompted to change the password for the `admin` user (or you can skip this step).

#### **3. Add Prometheus as a Data Source in Grafana**

After logging into Grafana, you need to configure Prometheus as the data source for Grafana to fetch metrics.

1. In Grafana's left sidebar, click on the gear icon (**Configuration**).
2. Click on **Data Sources**.
3. Click **Add data source**.
4. Choose **Prometheus** from the list of available data sources.
5. In the **HTTP** section, set the URL to `http://<MONITORING_SERVER_PRIVATE_IP>:9090`
6. Click **Save & Test** to verify that Grafana can connect to Prometheus.

![alt text](./images/image-24.png)

#### **4. Import Dashboards**

Finally, you can import pre-configured Grafana dashboards to visualize the metrics from PostgreSQL and PgBouncer.

1. In Grafana's left sidebar, click on the plus icon (**Create**) and select **Import**.
2. For PostgreSQL metrics, import the dashboard with ID [9628](https://grafana.com/grafana/dashboards/9628).
3. For PgBouncer metrics, import the dashboard with ID [13125](https://grafana.com/grafana/dashboards/13125).
4. Once the dashboards are imported, you should be able to visualize various metrics related to PostgreSQL and PgBouncer.


**PostgreSQL metrics**

![alt text](./images/image-25.png)


## Testing Concurrent connections

To test concurrent connections to your PostgreSQL database via PgBouncer, you can use tools like pgbench or custom scripts to simulate multiple clients connecting and executing queries simultaneously.

### **Grant Permissions to the User**

1. **Connect to PostgreSQL as a Superuser**:
   Connect to PostgreSQL using a superuser account (e.g., `postgres`):
   ```bash
   sudo -u postgres psql
   ```

2. **Grant Permissions**:
   Grant the necessary permissions to `myuser` for the `mydb` database:
   ```sql
   \c mydb
   GRANT ALL PRIVILEGES ON SCHEMA public TO myuser;
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO myuser;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO myuser;
   ```

3. **Exit PostgreSQL**:
   ```sql
   \q
   ```

### **Initialize pgbench**

Now that `myuser` has the necessary permissions, initialize the `pgbench` tables:

```bash
pgbench -h localhost -p 6432 -U myuser -i mydb
```
- Enter the password for `myuser` when prompted.

![alt text](./images/image-26.png)

### **Run the Benchmark**
Once the initialization is successful, you can run the benchmark to test concurrent connections:
```bash
pgbench -h localhost -p 6432 -U myuser -c 10 -j 2 -t 100 mydb
```
- `-c 10`: Number of clients (concurrent connections).
- `-j 2`: Number of threads.
- `-t 100`: Number of transactions per client.

### **Monitor Performance**
While running the benchmark, monitor the performance of PgBouncer and PostgreSQL using the Grafana dashboard or Prometheus metrics.

1. **Active Connections**:
   Check the number of active client and server connections:
   ```promql
   pgbouncer_pools_client_active_connections
   pgbouncer_pools_server_active_connections
   ```

   ![alt text](./images/image-28.png)

2. **Waiting Connections**:
   Check the number of clients waiting for a server connection:
   ```promql
   pgbouncer_pools_client_waiting_connections
   ```

3. **Query Throughput**:
   Check the rate of queries being executed:
   ```promql
   rate(pgbouncer_stats_queries_pooled_total[1m])
   ```
   ![alt text](./images/image-29.png)

4. **Transaction Throughput**:
   Check the rate of transactions being executed:
   ```promql
   rate(pgbouncer_stats_sql_transactions_pooled_total[1m])
   ```

### **Analyze Results**

After the benchmark completes, `pgbench` will output a summary of the results, including:
- **Transactions per second (TPS)**: The number of transactions completed per second.
- **Latency**: The average, minimum, and maximum latency for transactions.

![alt text](./images/image-27.png)

### **Increase Load**
To test higher loads, increase the number of clients (`-c`), threads (`-j`), and transactions (`-t`). For example:
```bash
pgbench -h localhost -p 6432 -U myuser -c 50 -j 4 -t 1000 mydb
```

### **Test with Custom Scripts**
If you want to test specific queries or workloads, you can create a custom SQL script for `pgbench`.

1. **Create a Script**:
   Create a file called `test.sql` with your custom queries:
   ```sql
   BEGIN;
   SELECT * FROM pgbench_accounts WHERE aid = 1;
   UPDATE pgbench_accounts SET abalance = abalance + 1 WHERE aid = 1;
   COMMIT;
   ```

2. **Run the Benchmark**:
   Run `pgbench` with the custom script:
   ```bash
   pgbench -h localhost -p 6432 -U myuser -c 10 -j 2 -t 100 -f test.sql mydb
   ```

## **Conclusion**  

PgBouncer is a powerful and lightweight connection pooler for PostgreSQL that optimizes connection management, reduces resource usage, and improves performance. By using proper configurations and pooling modes, you can ensure your application scales efficiently, even under high-concurrency loads. If you need **high availability**, you can combine PgBouncer with **Pgpool-II** or **HAProxy** to distribute connections across multiple database instances.