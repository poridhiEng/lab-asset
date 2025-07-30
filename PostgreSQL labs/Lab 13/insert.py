import psycopg2
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Database connection parameters
conn = psycopg2.connect(
    dbname="ecommerce", # change the database name to your own database name
    user="postgres",
    password="newpassword", # change the password to your own password
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Function to generate and insert users
def generate_users(num_users):
    used_emails = set()  # Track used emails to ensure uniqueness
    for _ in range(num_users):
        while True: # Repeat until an unique email is found
            name = fake.name()
            email = fake.email()
            if email not in used_emails:  # Ensure email is unique
                used_emails.add(email)
                break
        phone = fake.phone_number()[:20]  # Truncate to 20 characters
        address = fake.address()
        created_at = fake.date_time_between(start_date="-1y", end_date="now")
        cursor.execute(
            "INSERT INTO users (name, email, phone, address, created_at) VALUES (%s, %s, %s, %s, %s)",
            (name, email, phone, address, created_at)
        )
    conn.commit()
    print(f"Inserted {num_users} users.")

# Function to generate and insert products
def generate_products(num_products):
    for _ in range(num_products):
        name = fake.word().capitalize() + " " + fake.word().capitalize()
        description = fake.sentence()
        price = round(random.uniform(10, 1000), 2)
        stock_quantity = random.randint(10, 1000)
        created_at = fake.date_time_between(start_date="-1y", end_date="now")
        cursor.execute(
            "INSERT INTO products (name, description, price, stock_quantity, created_at) VALUES (%s, %s, %s, %s, %s)",
            (name, description, price, stock_quantity, created_at)
        )
    conn.commit()
    print(f"Inserted {num_products} products.")

# Function to generate and insert orders
def generate_orders(num_orders):
    cursor.execute("SELECT user_id FROM users")
    user_ids = [row[0] for row in cursor.fetchall()]
    for _ in range(num_orders):
        user_id = random.choice(user_ids)
        total_amount = round(random.uniform(10, 1000), 2)
        status = random.choice(['pending', 'completed', 'cancelled'])
        created_at = fake.date_time_between(start_date="-1y", end_date="now")
        cursor.execute(
            "INSERT INTO orders (user_id, total_amount, status, created_at) VALUES (%s, %s, %s, %s)",
            (user_id, total_amount, status, created_at)
        )
    conn.commit()
    print(f"Inserted {num_orders} orders.")

# Function to generate and insert order_items
def generate_order_items(num_items):
    cursor.execute("SELECT order_id FROM orders") # get the order_id from orders table
    order_ids = [row[0] for row in cursor.fetchall()]
    cursor.execute("SELECT product_id, price FROM products") # get the product_id, price from products table
    products = cursor.fetchall()
    for _ in range(num_items):
        order_id = random.choice(order_ids)
        product_id, price = random.choice(products)
        quantity = random.randint(1, 10)
        cursor.execute(
            "INSERT INTO order_items (order_id, product_id, quantity, price) VALUES (%s, %s, %s, %s)",
            (order_id, product_id, quantity, price)
        )
    conn.commit()
    print(f"Inserted {num_items} order items.")

# Function to generate and insert payments
def generate_payments(num_payments):
    cursor.execute("SELECT order_id FROM orders")
    order_ids = [row[0] for row in cursor.fetchall()]
    for _ in range(num_payments):
        order_id = random.choice(order_ids)
        payment_method = random.choice(['credit_card', 'paypal', 'bank_transfer'])
        payment_status = random.choice(['success', 'pending', 'failed'])
        transaction_id = fake.uuid4()
        created_at = fake.date_time_between(start_date="-1y", end_date="now")
        cursor.execute(
            "INSERT INTO payments (order_id, payment_method, payment_status, transaction_id, created_at) VALUES (%s, %s, %s, %s, %s)",
            (order_id, payment_method, payment_status, transaction_id, created_at)
        )
    conn.commit()
    print(f"Inserted {num_payments} payments.")

# Generate data
generate_users(1000)          # Insert 1000 users
generate_products(500)        # Insert 500 products
generate_orders(5000)         # Insert 5000 orders
generate_order_items(10000)   # Insert 10000 order items
generate_payments(5000)       # Insert 5000 payments

# Close the connection
cursor.close()
conn.close()