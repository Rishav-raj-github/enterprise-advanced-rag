import os
import sqlite3
import datetime
import random

def seed_database():
    db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "business_metrics.db")
    
    print(f"Creating and seeding SQLite database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Create Tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        region TEXT NOT NULL,
        signup_date TEXT NOT NULL
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        sale_date TEXT NOT NULL,
        total_amount REAL NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    """)
    
    # 2. Insert Products
    products_data = [
        ("Enterprise Server A", "Hardware", 1200.00),
        ("Developer Workstation Pro", "Hardware", 2500.00),
        ("Cloud Compute Core v1", "Cloud", 0.05), # price per hour
        ("Acme SaaS License Tier 1", "SaaS", 49.00),
        ("Acme SaaS License Enterprise", "SaaS", 499.00),
        ("Premium Consulting (Hour)", "Consulting", 250.00),
        ("Network Router X-500", "Hardware", 450.00),
        ("CyberSecurity Suite Basic", "Software", 199.00),
        ("AI Orchestration Suite", "Software", 899.00)
    ]
    cursor.executemany("INSERT INTO products (name, category, price) VALUES (?, ?, ?)", products_data)
    
    # 3. Insert Customers
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"]
    first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda", "William", "Elizabeth"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    
    customers_data = []
    for i in range(50):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        region = random.choice(regions)
        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2025, 12, 31)
        signup_date = start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days))
        customers_data.append((name, region, signup_date.isoformat()))
        
    cursor.executemany("INSERT INTO customers (name, region, signup_date) VALUES (?, ?, ?)", customers_data)
    
    # 4. Insert Sales (Transactions)
    sales_data = []
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2025, 12, 31)
    
    # We want a deterministic-looking database with around 200 sales
    random.seed(42) # Ensure seed reproducibility
    for _ in range(200):
        cust_id = random.randint(1, 50)
        prod_id = random.randint(1, len(products_data))
        qty = random.randint(1, 10)
        
        # Get price
        cursor.execute("SELECT price, category FROM products WHERE product_id = ?", (prod_id,))
        price, category = cursor.fetchone()
        
        # Calculate total amount
        if category == "Cloud":
            # Cloud has many hours
            qty = random.randint(100, 5000)
            
        total_amount = qty * price
        
        # Generate sale date
        sale_date = start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days))
        sales_data.append((cust_id, prod_id, qty, sale_date.isoformat(), round(total_amount, 2)))
        
    cursor.executemany("INSERT INTO sales (customer_id, product_id, quantity, sale_date, total_amount) VALUES (?, ?, ?, ?, ?)", sales_data)
    
    conn.commit()
    conn.close()
    print("Database seeding completed successfully!")

if __name__ == "__main__":
    seed_database()
