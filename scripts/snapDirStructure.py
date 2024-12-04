import os

def generate_dir_structure(dir_path, output_file):
    def tree(directory, prefix=""):
        entries = sorted(os.listdir(directory))
        structure = []
        for i, entry in enumerate(entries):
            path = os.path.join(directory, entry)
            connector = "└── " if i == len(entries) - 1 else "├── "
            structure.append(f"{prefix}{connector}{entry}")
            if os.path.isdir(path):
                extension = "    " if i == len(entries) - 1 else "│   "
                structure.extend(tree(path, prefix + extension))
        return structure

    try:
        if not os.path.exists(dir_path):
            print(f"Error: Directory '{dir_path}' does not exist.")
            return

        with open(output_file, "w") as f:
            f.write(f"Directory structure of: {dir_path}\n")
            f.write("\n".join(tree(dir_path)))
        
        print(f"Directory structure has been written to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_dir = r"C:\Users\Ben Gur\src\src\wubai\data"
    output_txt = r"C:\Users\Ben Gur\src\src\wubai\struct.txt"
    generate_dir_structure(input_dir, output_txt)
