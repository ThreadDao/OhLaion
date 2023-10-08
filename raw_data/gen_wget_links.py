def gen_links(base_url, data_type: str, index=0):
    if data_type == "metadata":
        url = f'{base_url}laion2B-multi-metadata/metadata_{index:04d}.parquet'
    elif data_type == "vector":
        url = f'{base_url}img_emb/img_emb_{index:04d}.npy'
    return url


def write_links(base_url, end: int, start=0):
    # write vector links
    with open('./wget_links/vector_links.txt', 'a') as f:
        for i in range(start, end):
            file_url = gen_links(base_url, data_type="vector", index=i)
            f.writelines(file_url + "\n")

    # write metadata links
    with open('./wget_links/metadata_links.txt', 'a') as f:
        for i in range(start, end):
            file_url = gen_links(base_url, data_type="metadata", index=i)
            f.writelines(file_url + "\n")


start = 0
end = 5
base_url = 'https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-multi/'
write_links(base_url, end=end, start=start)
