import os
import requests

TOKEN = "null"
LINKS_FILE = "links21.txt"
BASE_FOLDER = "bdc_data2"

def download_bdc_data(links_file, token):
    if not os.path.exists(BASE_FOLDER):
        os.makedirs(BASE_FOLDER)

    with open(links_file, 'r') as f:
        links = [line.strip() for line in f if line.strip()]

    for url in links:
        url_ready = url.replace("access_token=null", f"access_token={token}")
        
        #informações da estrutura da URL (muito mais seguro)
        # .../2026_04/CBERS_..._2026_04_28.../158_129_0/4_BC_.../arquivo.tif
        path_parts = url.split('/')
        filename = path_parts[-1].split('?')[0]
        
        try:
            # O Tile ID da URL no BDC
            # "158_129_0"
            tile_str = path_parts[-3] 
            
            # nome do arquivo: CBERS_4_AWFI_YYYYMMDD_...

            file_parts = filename.split('_')
            date_str = file_parts[3] 
        except (IndexError, ValueError):
            # Fallback
            tile_str = "unknown_tile"
            date_str = "unknown_date"

        # Cria a estrutura: bdc_data/TILE/DATA/
        target_dir = os.path.join(BASE_FOLDER, tile_str, date_str)
        os.makedirs(target_dir, exist_ok=True)
            
        target_file = os.path.join(target_dir, filename)

        #Faz o download
        if os.path.exists(target_file):
            print(f"[-] Arquivo já existe: {filename}")
            continue

        print(f"[+] Baixando: {filename}...")
        try:
            response = requests.get(url_ready, stream=True, timeout=30)
            response.raise_for_status()
            with open(target_file, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    out_file.write(chunk)
        except Exception as e:
            print(f"[!] Erro ao baixar {filename}: {e}")

if __name__ == "__main__":
    download_bdc_data(LINKS_FILE, TOKEN)
