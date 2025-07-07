import fitz  # PyMuPDF
import pandas as pd
import re
import os
from datetime import datetime

# AI categorization imports
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Hugging Face transformers not installed. Install with: pip install transformers torch")

# Global classifier variable to avoid reloading
classifier = None

def initialize_classifier():
    """Initialize the Hugging Face classifier (one-time setup)"""
    global classifier
    
    if not HF_AVAILABLE:
        print("❌ Hugging Face transformers not available. Skipping AI categorization.")
        return False
    
    if classifier is None:
        print("🤖 Loading AI model for transaction categorization...")
        print("📥 This may take a few minutes on first run (downloading ~500MB model)...")
        
        try:
            # Use zero-shot classification model
            classifier = pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli")
            print("✅ AI model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading AI model: {e}")
            return False
    
    return True

def categorize_transaction(description):
    """Categorize a transaction using hybrid approach: keywords first, then AI"""
    
    # First, try rule-based categorization for obvious cases
    rule_based_category = fallback_categorization(description)
    
    # If rule-based found a specific category (not Miscellaneous), use it
    if rule_based_category != "Miscellaneous":
        return rule_based_category
    
    # Otherwise, use AI for ambiguous cases
    global classifier
    
    if classifier is None:
        return "Miscellaneous"
    
    # Define categories for classification (using clean names directly)
    candidate_labels = [
        "Food & Dining",
        "Transportation", 
        "Shopping",
        "Entertainment",
        "Subscriptions",
        "Healthcare",
        "ATM/Cash",
        "Transfer",
        "Groceries"
    ]
    
    try:
        result = classifier(description, candidate_labels)
        
        # Get the top prediction and its confidence score
        top_category = result['labels'][0]
        top_confidence = result['scores'][0]
        
        # Only return the category if confidence is high enough
        # Otherwise return Miscellaneous
        confidence_threshold = 0.7  # Adjust this value (0.0 to 1.0)
        
        if top_confidence >= confidence_threshold:
            return top_category
        else:
            print(f"⚠️  Low confidence ({top_confidence:.2f}) for '{description}' → Miscellaneous")
            return "Miscellaneous"
        
    except Exception as e:
        print(f"⚠️  Error categorizing '{description}': {e}")
        return "Miscellaneous"

def fallback_categorization(description):
    """Enhanced rule-based categorization for obvious cases"""
    desc_lower = description.lower()
    
    categories = {
        'Food & Dining': [
            # Fast food chains
            'mcdonald', 'tim horton', 'burger king', 'kfc', 'subway', 'pizza',
            'starbucks', 'coffee', 'restaurant', 'cafe', 'diner', 'wendy',
            'taco bell', 'popeyes', 'a&w', 'dairy queen', 'harveys',
            # Food keywords
            'resto', 'bistro', 'grill', 'kitchen', 'bar & grill',
            # Your specific transactions
            'golden fish', 'arby', 'nuri village', 'yogurt', 'poke'
        ],
        'Transportation': [
            'uber', 'lyft', 'taxi', 'gas', 'petro', 'shell', 'esso', 'parking',
            'transit', 'go train', 'ttc', 'presto', 'via rail'
        ],
        'Shopping': [
            'amazon', 'walmart', 'target', 'costco', 'canadian tire', 'home depot',
            'loblaws', 'shoppers', 'best buy', 'future shop'
        ],
        'Entertainment': [
            'netflix', 'spotify', 'movie', 'cinema', 'theatre', 'concert',
            'waterloo star'  # seems like entertainment venue
        ],
        'Subscriptions': [
            'spotify', 'netflix', 'apple music', 'subscription', 'monthly fee',
            'gym membership', 'planet fitness'
        ],
        'Transfer': [
            'trsf', 'transfer', 'e-transfer', 'interac', 'payment to'
        ],
        'ATM/Cash': [
            'atm', 'cash withdrawal', 'bank machine'
        ],
        'Healthcare': [
            'pharmacy', 'shoppers drug', 'medical', 'doctor', 'hospital',
            'dental', 'clinic', 'health'
        ],
        'Groceries': [
            'loblaws', 'metro', 'sobeys', 'food basics', 'no frills',
            'supermarket', 'grocery', 'fresh'
        ],
        'Miscellaneous': [
            'hi yogurt'  # unclear what this is
        ]
    }
    
    # Check each category
    for category, keywords in categories.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    
    return 'Miscellaneous'

def add_categories_to_dataframe(df):
    """Add category column to the dataframe"""
    if len(df) == 0:
        return df
    
    # Initialize AI if available
    ai_available = initialize_classifier()
    
    print(f"🏷️  Categorizing {len(df)} transactions...")
    
    # Add categories to the dataframe
    if ai_available:
        print("🤖 Using hybrid categorization (keywords + AI)...")
        df['Category'] = df['Description'].apply(categorize_transaction)
    else:
        print("📝 Using rule-based categorization...")
        df['Category'] = df['Description'].apply(fallback_categorization)
    
    return df

def extract_bmo_transactions(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    # Focus on the transaction section
    if "Transactions since your last statement" not in full_text:
        raise ValueError("Could not find transaction section")
        
    txn_section = full_text.split("Transactions since your last statement", 1)[1]
    
    # Split by the subtotal to get just the transaction lines
    if "Subtotal for" in txn_section:
        txn_section = txn_section.split("Subtotal for", 1)[0]
    
    lines = txn_section.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    transactions = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Skip header lines and card info
        if (not line or "TRANS" in line or "DATE" in line or "DESCRIPTION" in line or 
            "AMOUNT" in line or "Card number:" in line or "ETHAN QY WANG" in line):
            i += 1
            continue
            
        # Look for lines that start with date pattern (Apr. 4 Apr. 7)
        date_match = re.match(r'^([A-Z][a-z]{2}\. \d{1,2})\s+([A-Z][a-z]{2}\. \d{1,2})$', line)
        if date_match:
            trans_date = date_match.group(1)  # Apr. 4
            post_date = date_match.group(2)   # Apr. 7
            
            # Next lines contain the merchant info and amount
            i += 1
            merchant_lines = []
            amount = None
            
            # Collect lines until we find the amount
            while i < len(lines):
                current_line = lines[i].strip()
                if not current_line:
                    i += 1
                    continue
                
                # Check if this line has an amount at the end
                amount_match = re.search(r'(\d+\.\d{2})(\s+CR)?$', current_line)
                if amount_match:
                    amount = float(amount_match.group(1))
                    if amount_match.group(2):  # CR means credit (negative)
                        amount = -amount
                    
                    # Add the part before the amount to merchant info
                    line_before_amount = current_line[:current_line.rfind(amount_match.group(0))].strip()
                    if line_before_amount:
                        merchant_lines.append(line_before_amount)
                    break
                else:
                    merchant_lines.append(current_line)
                    
                i += 1
            
            # Build the description
            if amount is not None:
                description = " ".join(merchant_lines)
                transactions.append([trans_date, post_date, description, amount])
        
        i += 1
    
    df = pd.DataFrame(transactions, columns=["Transaction Date", "Posted Date", "Description", "Amount"])
    
    # Add categories to the dataframe
    df = add_categories_to_dataframe(df)
    
    return df

def get_month_year_from_transactions(df):
    """Extract month/year from the first transaction for grouping"""
    if len(df) == 0:
        return None, None
    
    # Parse the first transaction date to get month/year
    first_date = df.iloc[0]["Transaction Date"]
    # Convert "Apr. 4" to datetime
    month_abbr = first_date.split('.')[0]
    
    if month_abbr:
        # Assume current year for now - you could extract from PDF if needed
        year = 2025
        return year, month_abbr
    
    return None, None

def append_to_log(new_df, log_file="bmo_transactions_log.xlsx"):
    """Append new transactions to existing log file, grouped by month"""
    
    if len(new_df) == 0:
        print("No new transactions to add")
        return
    
    year, month = get_month_year_from_transactions(new_df)
    
    # Check if log file exists
    if os.path.exists(log_file):
        # Read existing data
        existing_df = pd.read_excel(log_file)
        
        # Check if this month already exists
        if not existing_df.empty and "Year" in existing_df.columns and "Month" in existing_df.columns:
            # Remove existing entries for this month (in case of reprocessing)
            existing_df = existing_df[~((existing_df["Year"] == year) & (existing_df["Month"] == month))]
    else:
        existing_df = pd.DataFrame()
    
    # Add year and month columns to new transactions
    new_df_with_month = new_df.copy()
    new_df_with_month.insert(0, "Year", year)
    new_df_with_month.insert(1, "Month", month)
    
    # Combine old and new data
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df_with_month], ignore_index=True)
    else:
        combined_df = new_df_with_month
    
    # Sort by year and month
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    def sort_key(row):
        try:
            year_val = int(row["Year"])
            month_idx = month_order.index(row["Month"]) if row["Month"] in month_order else 99
            return (year_val, month_idx)
        except:
            return (9999, 99)  # Put unknown formats at the end
    
    combined_df['sort_key'] = combined_df.apply(sort_key, axis=1)
    combined_df = combined_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    # Save to file
    combined_df.to_excel(log_file, index=False)
    
    return combined_df, year, month

def main():
    pdf_file = "May 5, 2025.pdf"
    log_file = "bmo_transactions_log.xlsx"
    
    # Extract transactions from PDF
    df = extract_bmo_transactions(pdf_file)
    pd.options.display.float_format = '${:,.2f}'.format
    
    if len(df) == 0:
        print("❌ No transactions found. Let me debug...")
        # Debug output
        doc = fitz.open(pdf_file)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        txn_section = full_text.split("Transactions since your last statement", 1)[1]
        if "Subtotal for" in txn_section:
            txn_section = txn_section.split("Subtotal for", 1)[0]
        
        lines = txn_section.split('\n')
        print("Raw lines from transaction section:")
        for i, line in enumerate(lines[:20]):  # Show first 20 lines
            print(f"{i}: '{line.strip()}'")
    else:
        # Append to log
        combined_df, year, month = append_to_log(df, log_file)
        
        print(f"✅ Extracted {len(df)} transactions for {month} {year}")
        print(f"✅ Updated log file: {log_file}")
        print(f"✅ Total transactions in log: {len(combined_df)}")
        
        # Show summary by month
        if "Year" in combined_df.columns and "Month" in combined_df.columns:
            monthly_summary = combined_df.groupby(["Year", "Month"]).agg({
                "Amount": ["count", "sum"]
            }).round(2)
            monthly_summary.columns = ["Transaction Count", "Total Amount"]
            print("\n📊 Monthly Summary:")
            print(monthly_summary)
        
        # Show category breakdown for latest month
        if "Category" in df.columns:
            print(f"\n🏷️  Category breakdown for {month} {year}:")
            category_summary = df.groupby("Category").agg({
                "Amount": ["count", "sum"]
            }).round(2)
            category_summary.columns = ["Count", "Total Amount"]
            print(category_summary)
        
        print(f"\n📝 Latest transactions from {month} {year}:")
        print(df.head())

if __name__ == "__main__":
    main()