import pandas as pd
import random

def generate_synthetic_data():
    records = set() # Use a set to detect and prevent duplicates easily
    final_records = []
    
    # Helper to get unique descriptions
    def add_record(desc, hs, cat):
        # ensure unique
        while desc in records:
            # simple trick: append a random serial or ID to make it structurally unique if collision happens
            desc += f" ID-{random.randint(100, 9999)}"
        records.add(desc)
        final_records.append({
            "description": desc,
            "hs_code": hs,
            "category": cat
        })

    # Base 8517: 30 items
    models_8517 = ["iPhone 14 Pro", "Galaxy S23 Ultra", "Pixel 7 Pro", "OnePlus 11", "Xiaomi 13 Router", "Cisco Switch Catalyst", "TP-Link WiFi 6", "Nokia 3310 Retry", "Smartphone", "5G Network Hub"]
    attrs_8517 = ["256GB Black Unlocked", "GSM CDMA 5G 128GB", "Dual Band Gigabit", "24-Port Managed", "512GB Ceramic Shield", "refurbished factory", "new in box", "midnight color", "space gray"]
    for _ in range(30):
        add_record(f"{random.choice(models_8517)} {random.choice(attrs_8517)}", "8517", "Electronics/Networking")
        
    # 8525: 30 items
    models_8525 = ["Digital Camera setup with 4K video", "Action Camera waterproof 5.3K", "Security surveillance camera wireless", "Drone quadcopter with integrated high-definition broadcasting camera", "Mirrorless DSLR body only", "Digital streaming apparatus with built-in camera"]
    attrs_8525 = ["brand new", "bundle pack", "with smartphone wifi control", "night vision capable", "24.2 MP", "vlogging kit", "outdoor proof", "zoom lens included"]
    for _ in range(30):
        add_record(f"{random.choice(models_8525)} - {random.choice(attrs_8525)}", "8525", "Camera/Optical")

    # Apparel: 6109 (Knitted) vs 6205 (Woven) - 30 items each
    colors = ["Red", "Blue", "Green", "Black", "White", "Grey", "Navy", "Yellow", "Charcoal", "Olive"]
    sizes = ["S", "M", "L", "XL", "XXL", "XS"]
    
    # Knitted (6109)
    for _ in range(20):
        add_record(f"Men's 100% Cotton Crewneck T-Shirt Knitted short sleeve size {random.choice(sizes)} color {random.choice(colors)}", "6109", "Apparel (Knitted)")
    for _ in range(10): # Vague
        add_record(f"Unisex heavy cotton relaxed fit graphic tee {random.choice(colors)} size {random.choice(sizes)}", "6109", "Apparel (Vague)")

    # Woven (6205)
    for _ in range(20):
        add_record(f"Men's Long Sleeve Button-Down Oxford Dress Shirt woven size {random.choice(sizes)} color {random.choice(colors)}", "6205", "Apparel (Woven)")
    for _ in range(10): # Vague
        add_record(f"Cotton blend casual collar shirt short sleeve summer wear {random.choice(colors)}", "6205", "Apparel (Vague)")

    # Furniture (9403) - 30 items
    furnitures = ["Wooden dining table set with 4 chairs", "Metal frame office desk with wooden tabletop", "Large plastic outdoor garden storage box and bench", "Molded plastic chair for indoor outdoor use", "Rustic oak bookshelf 5-tier", "Ergonomic mesh office chair with wooden armrests"]
    furn_attrs = ["oak finish", "stackable", "weather resistant", "premium quality", "easy assemble", "dark walnut", "light brown", "industrial style"]
    for _ in range(30):
        add_record(f"{random.choice(furnitures)} - {random.choice(furn_attrs)}", "9403", "Furniture")

    # Plastics (3926) - 30 items
    plastics = ["Set of 5 plastic storage containers for kitchen, airtight", "Small plastic step stool for kids", "Molded plastic protective shell carrying case", "Plastic organizer box", "Clear plastic desk organizer for office supplies", "Heavy duty plastic storage bin"]
    plas_attrs = ["12 inches high", "medium size", "27 gallon", "BPA-free", "with snap lid", "multi-compartment", "impact resistant", "stacking design"]
    colors_plas = ["Clear", "Black", "White", "Blue", "Green", "Red", "Grey", "Transparent"]
    for _ in range(30):
        add_record(f"{random.choice(colors_plas)} {random.choice(plastics)} [{random.choice(plas_attrs)}]", "3926", "Plastics")
    
    df = pd.DataFrame(final_records)
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    output_path = "synthetic_customs_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic UNIQUE records to {output_path}")

if __name__ == "__main__":
    generate_synthetic_data()
