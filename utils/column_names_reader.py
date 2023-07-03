import sys
sys.path.append("../mutation_effect_analysis")

import gzip

def get_col_names_from_filehandle(file_handle, col_names_line_starting_symbol):
    for line_no, line in enumerate(file_handle):
        # print(line)
        if line.startswith(col_names_line_starting_symbol):
            line = line[1:] # skipping the # symbol
            line = line.rstrip()
            col_names = [x for x in line.split('\t')]
            # if line_no==100: break
            break
    return col_names

def get_col_names_from_textfile(filepath, col_names_line_starting_symbol):
    with open(filepath, "rt") as file_handle:
        return get_col_names_from_filehandle(file_handle, col_names_line_starting_symbol)

def get_col_names_from_gzip(filepath, col_names_line_starting_symbol):
    with gzip.open(filepath, "rt") as file_handle:
        return get_col_names_from_filehandle(file_handle, col_names_line_starting_symbol)
        

def get_col_names(filepath:str, col_names_line_starting_symbol:str):        
    if filepath.endswith(".txt"): 
        return get_col_names_from_textfile(filepath, col_names_line_starting_symbol)
    elif filepath.endswith(".gz"):
        return get_col_names_from_gzip(filepath, col_names_line_starting_symbol)
    else:
        raise NotImplementedError("The fileformat is not supported.")

# example usage
# print(get_col_names("data/clinvar/2023_01/hgvs4variation.txt.gz", "#Symbol"))
# print(get_col_names("data/snp_population_freq/2020_11.vcf.gz", "#CHROM"))