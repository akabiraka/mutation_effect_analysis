

all_method_names = ['sift', 'polyphen2_HVAR', 'metarnn', 'revel', 'mvp', 'cadd_raw', 'integrated_fitCons', 'phyloP17way_primate', 'phastCons17way_primate', 'bStatistic', 
                'esm1b_t33_650M_UR50S', 'esm1v_t33_650M_UR90S', 'esm2_t33_650M_UR50D', 
                'prottrans_bert_bfd', 'prottrans_albert_bfd', 'plus_rnn', 'prottrans_t5_xl_u50', 
                'vespa', 'vespal', 
                'proteinbert', 'sequnet', 'protbert', 'unirep', 'conservation']

selected_method_names = ['sift', 'polyphen2_HVAR', 'revel', 'cadd_raw', 'phyloP17way_primate', 'phastCons17way_primate', 
                'metarnn',  'mvp', 'sequnet', 'vespa',
                'esm1b_t33_650M_UR50S', 'esm1v_t33_650M_UR90S', 'esm2_t33_650M_UR50D', 
                'prottrans_bert_bfd', 'prottrans_t5_xl_u50', 'proteinbert', 'protbert', 'unirep']
# not considered anymore: 'random_classifier', 'integrated_fitCons', 'bStatistic', 'conservation' 


methods_smaller_means_damaging_from_paper = ['sift'] # from the paper
# from popu_freq_Rare_vs_Common th-max negative
all_methods_smaller_means_damaging = ['sift', 'vespa', 'esm1b_t33_650M_UR50S', 'esm1v_t33_650M_UR90S', 'esm2_t33_650M_UR50D', 'prottrans_bert_bfd', 'prottrans_t5_xl_u50', 'proteinbert']



pmd_colors_dict = {"Knock-out":"red", "Effect":"darkred", "No-effect":"darkblue"}
pmd_colors_list = [v for k, v in pmd_colors_dict.items()]
pmd_class_order = [k for k, v in pmd_colors_dict.items()]

patho_colors_dict = {"Pathogenic":"coral", "Likely-pathogenic":"lightcoral", "Neutral":"gray"}
patho_colors_list = [v for k, v in patho_colors_dict.items()]
patho_class_order = [k for k, v in patho_colors_dict.items()]

popu_freq_colors_dict = {"Singleton":"maroon", "Ultra-rare":"chocolate", "Rare":"orange", "Common":"lightsteelblue"}
popu_freq_colors_list = [v for k, v in popu_freq_colors_dict.items()]
popu_freq_class_order = [k for k, v in popu_freq_colors_dict.items()]
popu_freq_class_order_wo_singleton = ["Ultra-rare", "Rare", "Common"]