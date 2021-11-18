import os
import sox
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from scipy.stats import uniform
import matplotlib.pyplot as plt
from models import deeptone_classifier
from sklearn.naive_bayes import GaussianNB
import sklearn.gaussian_process.kernels as kerns
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

try:
    import deeptone
except Exception as e:
    raise Warning(e)

plt.style.use('fivethirtyeight')

def get_audio_files(data_folder, expected_sr, remove_unexpected_sr = False):
    # Find audio files 
    audio_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.wav')]
        
    # Removal of unxepected sampling rates
    unexptected_srs = []
    for audio_file in audio_files:
        if sox.file_info.sample_rate(os.path.join(data_folder,audio_file)) != expected_sr:
            unexptected_srs.append(audio_file)
    if len(unexptected_srs) > 0 & remove_unexpected_sr:
        audio_files = [i for i in audio_files if i not in  unexptected_srs]
        Warning(f'Found files with unxepcted sample rates: Removing {remove_unexpected_sr}')
        
    return audio_files

def get_metadata(audio_files, data_folder):         

    df_ = deeptone_classifier.create_metadata(audio_files = audio_files)  
    df_index = pd.read_csv(data_folder+"/Orestes_call-combo-classification.csv")
    df_index["file_name"] = [f + ".wav" for f in df_index.File.values]
    remove = df_index[df_index["type"]=="xx"].file_name.values
    df_index_keep = df_index[df_index["type"]!="xx"]
    df_index_keep["type"] = ["phc" if x == "climax" else "phi" for x in  df_index_keep.type]
    df_ = df_[~df_["file_name"].isin(remove)]
    df_.loc[df_.file_name.isin(df_index_keep.file_name).values, ['call_type']] = df_index_keep['type'].values

    # Subset data by removing pant grunt calls and modify labelling of combinations calls 
    dfi_keep = df_ 
    dfi_keep = dfi_keep.iloc[np.nonzero(dfi_keep.call_type.values != 'pg')[0],:]
    dfi_keep.loc[:,"call_type"] = ["Combination" if "-" in f else f for f in dfi_keep.call_type.values]

    try:
        os.system(f'mkdir {data_folder}/keep')
    except Exception as e:
        print(e)

    [os.system(f'cp {data_folder}/{f} {data_folder}/keep/{f}') for f in dfi_keep.file_name.values] 

    dfi_keep['call_type'] = dfi_keep['call_type'].str.replace('phi', "Pant-hoot-intro")
    dfi_keep['call_type'] = dfi_keep['call_type'].str.replace('phc', "Pant-hoot-climax")

    return dfi_keep

def analysis(data_folder, results_folder, front_end_dictionary, model_dictionary, C_choice_list, train_num_list, active_param_sampling, label, nested_labels):
    
    try:
        os.mkdir(results_folder)
    except Exception as e:
        print(e)
    
    root_seed = 1
     
    audio_files = get_audio_files(data_folder, expected_sr = 16000, remove_unexpected_sr = True)

    if len(audio_files)==0:
        raise EnvironmentError("No audio files found")

    with_plots = False

    reportz = pd.DataFrame()

    for name in list(front_end_dictionary.keys()): 

        front_end = front_end_dictionary[name]   

        # Create metadata from list of target audio files (information enclosed in filename)
        dfi_keep = get_metadata(audio_files, data_folder)

        # Get raw embeddings for each audio file and save as dictionary
        dfs = deeptone_classifier.get_embeddings(df = dfi_keep,
                            data_folder = data_folder,
                            normalise = True,
                            average= False,
                            expected_sr = 16000,
                            target_model = front_end["model_type"])
   
        # Transform the embeddings space by averaging over time
        embeddings = deeptone_classifier.transform_embeddings_space(dfs = dfs, average = True)
        embeddings.reset_index(drop=True, inplace=True)
        dfi_keep.reset_index(drop=True, inplace=True)
        dfi_keep = embeddings.merge(dfi_keep)
        df_pre_scores_all = pd.DataFrame(columns=["y_test_a","y_pred_a","iteration_id","classifier_id","call_type"]) 

        for C_choice in C_choice_list:
            
            for key in list(model_dictionary.keys()): 

                model_dictionary[key]['search_parameters'] = dict()

                if active_param_sampling:
                    param_sample = ["C",uniform(loc=C_choice+0.001,scale=(C_choice+0.001)*10)]
                else:
                    param_sample = None

                df_pre_scores = pd.DataFrame()
                
            
                for train_num in tqdm.tqdm(train_num_list): 

                    # Run multiple classification experiments for desired classifier
                    df_pre_scores, est_vals, n_train = deeptone_classifier.run_classifier(df = dfi_keep, 
                                                        train_num = train_num,
                                                        n_rand = 500,
                                                        seed = root_seed,
                                                        model_pack = model_dictionary[key],
                                                        label = label,
                                                        list_nested_label = nested_labels,
                                                        param_sample = param_sample)

                    if C_choice_list == [1]:
                        df_pre_scores_all = df_pre_scores_all.append(df_pre_scores)

                    # Plot confusion matrices for each classifiers by looping over the types of nested labels.
                    if with_plots:

                        df_pre_scores.loc[:,"Correct"] = df_pre_scores.y_test_a.values == df_pre_scores.y_pred_a.values

                        for nested in nested_labels:
                            df_pre_scores.loc[:,f"{nested}_a"] = [df_pre_scores.loc[:,nested].values[f] if df_pre_scores.Correct.values[f] == True else f'Wrong {df_pre_scores.y_pred_a.values[f]}' for f in list(range(df_pre_scores.shape[0]))]                                         

                        all_scores_chimps  = deeptone_classifier.average_metrics(df_pre_scores = df_pre_scores, 
                                                            y_test = "y_test_a", 
                                                            y_pred = "y_pred_a")

                        # Create title for plots
                        sub_results_folder = f"{results_folder}/{front_end['model_name']}_{key}_{train_num}"

                        deeptone_classifier.plot_averaged_scores(all_scores = all_scores_chimps, 
                                            title = sub_results_folder ,
                                            data_folder = data_folder)

                        for lab in nested_labels:

                            deeptone_classifier.plot_confusion_matrix_subplot(df = df_pre_scores,
                                                y_test = lab,
                                                y_pred = f"{lab}_a",
                                                title = sub_results_folder,
                                                data_folder = data_folder)

                    report_melt = pd.DataFrame(data = {'f1-score':est_vals['f1-score'],
                                                    'accuracy':est_vals['accuracy'],
                                                    'precision':est_vals['precision']
                    })

                    report_melt['train_num'] = train_num
                    report_melt['model_type'] = name
                    report_melt['C_choice'] = C_choice
                    
                    report = reportz.append(report_melt)

        if C_choice_list == [1]:
            _ = deeptone_classifier.get_aggregated_scores(df_pre_scores = df_pre_scores_all, 
                                y_test = "y_test_a", 
                                y_pred = "y_pred_a",
                                title = f'{results_folder}/{label}_classifier_comp',
                                data_folder = None,
                                with_plot = True)

            all_scores_chimps = deeptone_classifier.average_metrics(df_pre_scores = df_pre_scores_all, 
                                                y_test = "y_test_a", 
                                                y_pred = "y_pred_a")

            deeptone_classifier.plot_averaged_scores(all_scores = all_scores_chimps, 
                                title = f'{results_folder}/{label}_classifier_comp',
                                data_folder = data_folder)

            df_pre_scores_all.to_csv(f"{results_folder}/{label}_classifier_comp.csv")    

    return report

def save_and_plot_mfcc_vs_deeptone_comparisson(report):
    report.to_csv(f"{results_folder}/reportz_mfcc_v_identity.csv")

    type_all_m = pd.DataFrame( data={"model_type":["Identity","mfcc"],
                                    "Model Type":["Identity","MFCC"]}
    )
    report = pd.merge(report,type_all_m,on = "model_type")
    report_m = report.copy()
    report_m.reset_index(inplace=True) 
    report_m['train_num'] = [round(f,3) for f in report_m.train_num.values]
    plt.close()
    fig, ax = plt.subplots(figsize=[12,7])
    a = sns.lineplot(data=report_m, x="train_num", y="accuracy",hue = 'Model Type',ax=ax)
    a.set(xlabel = "Number of training points",ylabel = "Accuracy")
    plt.tight_layout()
    plt.savefig(f'{results_folder}/comparisson_deeptone_VS_mfcc.png') 

def manuscript_analysis():
    
    data_folder = "data/experiment_data"
    
    results_folder = "results_folder" 

    ############################################
    # Chimpanzee identity nested in call types # 
    ############################################

    _ = analysis(data_folder,
             results_folder,
             front_end_dictionary = {
                    "Identity":{
                        "model_type":deeptone.models.Identity.predict,
                        "model_name":"Identity"
                    }
                                    },
             model_dictionary = {
                    'NaiveBayes':{
                        'name' : 'Naive Bayes',
                        'fixed_parameters': None,
                        'model': GaussianNB(),
                        'search_parameters': None},
                    'RandomForest':{
                        'name' : 'Random Forest',
                        'fixed_parameters': None,
                        'model': RandomForestClassifier(),
                        'search_parameters': dict(n_estimators=np.arange(1,101))},
                    'GaussianProcess':{
                        'name' : 'Gaussian Processes',
                        'fixed_parameters': dict(kernel=kerns.RationalQuadratic()),
                        'model': GaussianProcessClassifier,
                        'search_parameters': None},
                    'SupportVectorMachine':{
                        'name' : 'Support Vector Machine',
                        'fixed_parameters': dict(kernel = 'rbf',
                                                decision_function_shape = 'ovo',
                                                gamma='scale'),
                        'model': svm.SVC,
                        'search_parameters': dict(C=uniform(loc=1, scale=30), degree = list(range(20)))} 
                                 },
             C_choice_list = [1],
             train_num_list = [0.85],
             active_param_sampling = False,
             label= "label",
             nested_labels= ["call_type"])


    ############################################
    # Call types nested in chimpanzee identity # 
    ############################################

    _ = analysis(data_folder,
            results_folder,
            front_end_dictionary = {
                    "Identity":{
                        "model_type":deeptone.models.Identity.predict,
                        "model_name":"Identity"
                    }
            },
            model_dictionary = {
                    'NaiveBayes':{
                        'name' : 'Naive Bayes',
                        'fixed_parameters': None,
                        'model': GaussianNB(),
                        'search_parameters': None},
                    'RandomForest':{
                        'name' : 'Random Forest',
                        'fixed_parameters': None,
                        'model': RandomForestClassifier(),
                        'search_parameters': dict(n_estimators=np.arange(1,101))},
                    'GaussianProcess':{
                        'name' : 'Gaussian Processes',
                        'fixed_parameters': dict(kernel=kerns.RationalQuadratic()),
                        'model': GaussianProcessClassifier,
                        'search_parameters': None},
                    'SupportVectorMachine':{
                        'name' : 'Support Vector Machine',
                        'fixed_parameters': dict(kernel = 'rbf',
                                                decision_function_shape = 'ovo',
                                                gamma='scale'),
                        'model': svm.SVC,
                        'search_parameters': dict(C=uniform(loc=1, scale=30), degree = list(range(20)))} 
             },
             C_choice_list = [1],
             train_num_list = [0.85],
             active_param_sampling = False,
             label= "call_type",
             nested_labels= ["label"])


    ##########################
    # MFCC vs DEEPTONE (SVM) #
    ##########################

    report = analysis(data_folder,
            results_folder,
            front_end_dictionary = {
                    "mfcc":{
                        "model_type":"mfcc",
                        "model_name":"mfcc"
                    },
                    "Identity":{
                        "model_type":deeptone.models.Identity.predict,
                        "model_name":"Identity"
                    }
            },
            model_dictionary = {
                    'SupportVectorMachine':{
                        'name' : 'Support Vector Machine',
                        'fixed_parameters' : dict(kernel = 'rbf',
                                                decision_function_shape = 'ovo',
                                                gamma='scale',
                                                C = 1),
                        'model' : svm.SVC
                    }                                  
            },
            C_choice_list= [0.1,1,10,100],
            train_num_list= np.arange(5,175,5),
            active_param_sampling= True,
            label=  "label",
            nested_labels= ['call_type'])

    save_and_plot_mfcc_vs_deeptone_comparisson(report)

def example_mfcc_analysis():
    
    data_folder = "data/experiment_data"
    
    results_folder = "results_folder" 

    #############
    # MFCC only #
    #############

    _ = analysis(data_folder,
             results_folder,
             front_end_dictionary = {
                    "mfcc":{
                        "model_type":"mfcc",
                        "model_name":"mfcc"
                    },
                                    },
             model_dictionary = {
                    'NaiveBayes':{
                        'name' : 'Naive Bayes',
                        'fixed_parameters': None,
                        'model': GaussianNB(),
                        'search_parameters': None},
                    'RandomForest':{
                        'name' : 'Random Forest',
                        'fixed_parameters': None,
                        'model': RandomForestClassifier(),
                        'search_parameters': dict(n_estimators=np.arange(1,101))},
                    'GaussianProcess':{
                        'name' : 'Gaussian Processes',
                        'fixed_parameters': dict(kernel=kerns.RationalQuadratic()),
                        'model': GaussianProcessClassifier,
                        'search_parameters': None},
                    'SupportVectorMachine':{
                        'name' : 'Support Vector Machine',
                        'fixed_parameters': dict(kernel = 'rbf',
                                                decision_function_shape = 'ovo',
                                                gamma='scale'),
                        'model': svm.SVC,
                        'search_parameters': dict(C=uniform(loc=1, scale=30), degree = list(range(20)))} 
                                 },
             C_choice_list = [1],
             train_num_list = [0.85],
             active_param_sampling = False,
             label= "label",
             nested_labels= ["call_type"])

if __name__ == '__main__':
    example_mfcc_analysis()


