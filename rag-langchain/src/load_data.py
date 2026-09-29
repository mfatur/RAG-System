import pandas as pd


# 1. Load raw CSV
df = pd.read_csv(
    "data/customer_support_tickets.csv",
    keep_default_na=False
)
print("Jumlah data awal:", len(df))


# 2. Pilih kolom yang relevan untuk RAG
rag_columns = [
    "Ticket ID",
    "Product Purchased",
    "Ticket Type",
    "Ticket Subject",
    "Ticket Description",
    "Ticket Status",
    "Resolution",
    "Ticket Priority",
    "Ticket Channel",
]

df = df[rag_columns]


# 3. Hapus baris yang deskripsinya kosong atau hanya berisi spasi
df = df[df["Ticket Description"].str.strip().ne("")]


# 4. Bersihkan whitespace pada kolom teks
text_columns = [
    "Product Purchased",
    "Ticket Type",
    "Ticket Subject",
    "Ticket Description",
    "Ticket Status",
    "Resolution",
    "Ticket Priority",
    "Ticket Channel",
]

for column in text_columns:
    df[column] = df[column].fillna("").astype(str).str.strip()
    
# Ganti placeholder nama produk dengan nilai kolom Product Purchased
df["Ticket Description"] = df.apply(
    lambda row: row["Ticket Description"].replace(
        "{product_purchased}",
        row["Product Purchased"]
    ),
    axis=1
)
# 5. Simpan hasil cleaning
df.to_csv("data/clean_tickets.csv", index=False)


print("Jumlah data setelah cleaning:", len(df))
print("\nKolom yang digunakan:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFile berhasil disimpan:")
print("data/clean_tickets.csv")