import fitz  # PyMuPDF
import pandas as pd
import re
import os
from datetime import datetime

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
    pdf_file = "June 5, 2025.pdf"
    log_file = "bmo_transactions_log.xlsx"
    
    # Extract transactions from PDF
    df = extract_bmo_transactions(pdf_file)
    
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
        
        print(f"\n📝 Latest transactions from {month} {year}:")
        print(df.head())

if __name__ == "__main__":
    main()