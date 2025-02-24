import os

def get_file_position(directory, filename):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    files.sort()
    for file in files:
        if file == filename:
            return files.index(file) + 1
        print(file)
    
    try:
        position = files.index(filename) + 1
        return position
    except ValueError:
        return None

if __name__ == "__main__":
    directory = "imgs/paths/3_agents/70.0_bgt/"
    filename = "1740399463.3915021.png"
    
    position = get_file_position(directory, filename)
    
    if position is not None:
        print(f"O arquivo '{filename}' está na posição {position} na ordem.")
    else:
        print(f"O arquivo '{filename}' não foi encontrado no diretório.")