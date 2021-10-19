import os
import tqdm
import random
import librosa
import deeptone
import numpy as np
import pandas as pd
import seaborn as sns
from pydub import AudioSegment
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, recall_score, accuracy_score, confusion_matrix, f1_score

global conv_fact
conv_fact = 2 ** 31


def acronym_gen(phrase):
    """Acronym generator

    Args:
        phrase (str): Phrase to shorten to acronym

    Returns:
        str: Acronym
    """    

    phrase_split = phrase.split()
    acronym = ""
    ## iterate through every substring
    for i in phrase_split:
        acronym = acronym + i[0].upper()
    return acronym


def get_model_predictions(x):
    """Transforms amplitudional time-series of sound file into Deeptone Identity embeddings.

    Args:
        x (float): Array containing amplutiodnal time-series of an audio file of length L (seconds) with 16kHz sample rate

    Returns:
        pd.Dataframe: Output of Deeptone Identity model (N x 128 dimensions)
    """    
    # Use the sampled (at 16kHz) amplitude time-series of the sound file as input in DeepTone models
    # to obtain the embedding representations of the latent space for each sample.
    # One vector will be generated for every 64 milliseconds
    identity = deeptone.models.Identity.predict(x)[0]

    # Create a dimensional index for each embedding vector
    d_index = list(range(len(identity.shape[0]))) 

    # Save each vector embedding as a coloumn, indexed (row-wise) by the time signature of each vector embedding elements
    df_ = pd.DataFrame(index = d_index)
    for i in range(identity.shape[1]):
        df_[f'identity_{i}'] = identity[:,i]

    return df_


def import_audio(audio_path, expected_sr, target_sr):
    """ Import audio from .wav file and slow down (optionak)

    Args:
        audio_path (str): Path to audio file
        expected_sr (int): Expected sample rate of audio
        target_sr (int): Target sample rate of audio
        
    Returns:
        [AudioSegment]: Audio file as a pydbu.AudioSegment object
    """    

    # Import audio using pydub
    sound = AudioSegment.from_wav(audio_path)
    
    # Check sample rate of file
    actual_sr = sound.frame_rate
    if actual_sr!=expected_sr:

        os.system(f'echo {audio_path} {actual_sr} >> wrong_sr.txt')
        sound = sound.set_frame_rate(expected_sr)
        Warning(f'Sample rate of {audio_path} not as expected ({actual_sr} instead of {expected_sr})')


    # Change playback speed of file
    slow_trimmed_sound = down_sample(sound, target_sr) 

    return slow_trimmed_sound


def down_sample(sound, sample_rate_out):
    """Downsample audio
    Args:
        sound (pydub.AudioSegment): Sound imported as a AudioSegment object
        sample_rate_out (int): Final sample rate for output data

    Returns:
        pydub.AudioSegment: Downsampled audio
    """    

    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate)
    })

    # Convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. 
    return sound_with_altered_frame_rate.set_frame_rate(sample_rate_out)


def create_metadata(audio_files):
    """ Create standardised metadata for reading files

    Args:
        audio_files (list): List indicating the relative path to target audio files

    Returns:
        pd.Dataframe: Dataframe containing standardised information for analysis
    """    
    
    # Pre-assign storage
    df_ = pd.DataFrame(data=audio_files,columns=["path_to_file"])
    df_['file_name'] = [os.path.split(i)[1] for i in df_.path_to_file.values]
    # Loop through audio files
    for row in tqdm.tqdm(df_.iterrows()):
        info_row = row[1].file_name.replace("(","_").replace(")","_").replace(".wav","_.wav").replace("__.wav","_.wav").split("_")
        df_.loc[row[0], "label"] = info_row[0]
        df_.loc[row[0], "call_type"] = info_row[1]
        df_.loc[row[0], "quality"] = info_row[-2]

    df_= df_.set_index('file_name',drop=False)
    df_.loc[:, 'label_idx'] = df_.label.astype('category').cat.codes

    return df_


def get_embeddings(df,data_folder,normalise,average,expected_sr,target_model):
    """ Obtain the Deeptone Identity embedding space (N x 128 dimensions) from audio files listed in df_

    Args:
        df (pd.DataFrame): Dataframe containing N x 128 dimensional embedding space
        data_folder (str): String containing path to data folder 
        normalise (boolean): Boolean dscribing whether data should be normalised or not
        play_back_factor (float): Facctor (of original playback speed) by which to slow down audio
        expected_sr (int): Expected sample rate of audio

    Returns:
        df (pd.Dataframe): Updated inputted datadrame that sqaushes Deeptone Identity embedding to string 
        X_ (np.array): Squashed (mean) embedding space of each file
        dfs (dict): Dictionary indexed by audio_file name containing full embedding space of each audio file.
    """    
    
    # Pre-assign storage
    dfs = dict()

    # Loop through audio files
    for row in df.iterrows():

        # Import audio files, ensure sample rate and normalise the amplitudional time-series
        if target_model == "mfcc":
            slow_trimmed_sound = import_audio(os.path.join(data_folder, row[1].path_to_file), expected_sr, 44100)
        else:
            slow_trimmed_sound = import_audio(os.path.join(data_folder, row[1].path_to_file), expected_sr, 16000)

        # Convert data stored in AudioSegment object from 16 bit signed integer to 32 bit floating point.
        x = np.array(slow_trimmed_sound.get_array_of_samples())/conv_fact

        if normalise == True: 
            x = x/max(x)

        if target_model == "mfcc":
            identity_file = np.transpose(librosa.feature.mfcc(y=x, sr=44100, n_mfcc=128,fmin=50,fmax=44100/2))
        else: 
            identity_file = target_model(x.reshape(1,len(x),1))[0,:,:]

        if average == True:
            identity_file = np.mean(identity_file,axis=0)

        # Storage
        dfs.update({row[1].path_to_file: identity_file})  
    
    return dfs


def transform_embeddings_space(dfs,average): 

    """Transorm N x 128 (i x j) dimensional embdedding space into 1 x 128 dimensions by average over (time) dimension "i". Save and output into one data frame.

    Args:
        df (pd.Data_frame): Dataframe containing audio files to be used for analysis
        dfs (dict): Dictionary (indexed by file name "i") cotaining pd.Dataframe of N_i x 128 embedding space of "i".

    Returns:
        (pd.Dataframe): DataFrame, indexed by "i", with average (over time) embedding space.
    """    

    # Calculates embeddings and averages over time
    audio_files = list(dfs.keys())

    n_features = dfs[audio_files[1]].shape[1]

    if average:
        embeddings = np.ndarray((len(audio_files), n_features))
        for idx, audio_file in enumerate(audio_files):
            embeddings[idx,:] = dfs[audio_file].mean(axis=0)

        embeddings = pd.DataFrame(data = embeddings, 
                            columns = [f'dim_{f}' for f in list(range(n_features))],
                            index = audio_files)
        embeddings["path_to_file"] = audio_files
    else:
        embeddings = pd.DataFrame(columns = [f'dim_{f}' for f in list(range(n_features))])
        for idx, audio_file in enumerate(audio_files):
            embeddings_temp = pd.DataFrame(data = dfs[audio_file],
                                        columns = [f'dim_{f}' for f in list(range(n_features))],
                                        index = [audio_file+str(f) for f in range(dfs[audio_file].shape[0])])
            embeddings_temp["path_to_file"] = audio_file
            embeddings = pd.concat([embeddings,embeddings_temp])
    return embeddings


def plot_count_variable_combo(target,nested,df,data_folder):
    """Count plot of nested variable in target one

    Args:
        target (str): Column name of main variable to be plotted.
        nested (str): Column name of secondary variable to be nested within "target".
        df (pd.DataFrame): Data to be plotted.
        data_folder (str): Location to save figures.
    """    
    df = df.rename(columns={target:"Chimp individual",nested:"Call type"})
    fig_dims = (15, 10)
    fig, ax = plt.subplots(figsize=fig_dims)
    sns.set_theme(style="whitegrid")
    g=sns.histplot(df, x="Chimp individual", hue="Call type", multiple="stack",ax=ax).set(xlabel=None)
    plt.xticks(rotation=90,size=20)
    plt.yticks(rotation=90,size=20)
    plt.ylabel("Count",size=20)
    plt.subplots_adjust(top=0.85,bottom=0.2)
    plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='20') # for legend title
    fig.savefig(f'{data_folder}/Chimps_call_type_X_ID.pdf',format= "pdf")
    fig.savefig(f'{data_folder}/Chimps_call_type_X_ID.png')
    

def run_classifier(df,train_num,n_rand,seed,model_pack,label,list_nested_label,param_sample=None):
    """ Apply a desired classifier to data.

    Args:
        df (pd.DataFrame): Dataframe indexed by data unit, whose columns contain, labels, other metadata and identity vector dimensions
        train_num (int,float): If integer, controls the number of samples to train on, if float, controls the proportion of sampled to train on
        n_rand (int): Number of times to run classifier. Each run randomly allocates the test and training data.
        seed (int): Random seed number.
        model_pack (dict): Dictionary contain the 'name' (a string), 'model' (passed as a function), "fixed_parameter" (kwargs, optional),  "fixed_parameter" (kwargs, optional)
        label (str): Points to column which to be trained on.
        list_nesed_labels (list,str): List of nested columns, relative to label.
        param_sample (list): First element should contain the parameters to be sampled on, second element the distribution to sample from. Defaults to None

    Returns:
        [list]: A pd.Dataframe containg aggregated predicted and corresponding test labels, the iteration id and the classifier id. Another dataframe containing overall classification scores

    """    

    
    random.seed(seed)
    

    states = random.sample(list(np.arange(1,10000)),n_rand)
    
    # Configure classifier options
    if param_sample:
        param, dist = param_sample
    indexes = list(df.index)
    model = model_pack["model"]
    search_parameters = model_pack["search_parameters"]
    fixed_parameters = model_pack["fixed_parameters"]
    overall_valls = dict()
    y_test_a = np.array([])
    y_pred_a = np.array([])
    y_nested_test = dict()
    if list_nested_label:
        for nested in list_nested_label:
            y_nested_test.update({nested:np.array([])})
    iteration_id = np.array([])
    classifier_id = np.array([])
    rep_idx = -1
    numb_labels = len(set(df.loc[:,label]))

    # Prepare dictionary
    if search_parameters:
        est_vals = dict.fromkeys(search_parameters)
        est_vals_keys = list(est_vals.keys())
        for est_val_key in est_vals.keys():
            est_vals[est_val_key] = np.zeros(n_rand)
        overall_valls.update({"est_vals":est_vals})
        overall_valls.update({"f1-score":np.zeros(n_rand)})
        overall_valls.update({"accuracy":np.zeros(n_rand)})
        overall_valls.update({"precision":np.zeros(n_rand)})
        overall_valls.update({"state":np.zeros(n_rand)})
    else: 
        overall_valls = {"f1-score":np.zeros(n_rand)}
        overall_valls.update({"accuracy":np.zeros(n_rand)})
        overall_valls.update({"precision":np.zeros(n_rand)})
        overall_valls.update({"state":np.zeros(n_rand)})

    # Get embeddings
    embedding_cols = [col for col in df.columns if 'dim' in col]

    # Loop through classifier runs
    for state in states:
        rep_idx+=1

        if param_sample:

            fixed_parameters.update({param:dist.rvs(1)})

        # Get training and test set
        while True:

            if train_num >1.1:
                index_train, index_test = train_test_split(list(set(indexes)), test_size = len(list(set(indexes))) - train_num)
            else:
                index_train, index_test = train_test_split(list(set(indexes)), test_size = 1 - train_num)

            X_train = np.array(df.loc[index_train,embedding_cols])
            X_test = np.array(df.loc[index_test,embedding_cols])
            y_train = np.array(df.loc[index_train,label])
            y_test = np.array(df.loc[index_test,label])
            if len(list(set(y_train)))>=numb_labels:
                break

        X = np.array(df.loc[:,embedding_cols])
        y = np.array(df.loc[:,label])

        # Get search and fixed parameters
        if search_parameters:

            if fixed_parameters:
                # Optimise parameters with randomised search with certain fixed parameters
                optimisation_algorithm = RandomizedSearchCV(model(**fixed_parameters), search_parameters, n_jobs=-1,cv=2) #,n_iter=100
            else:
                # Optimise parameters with randomised search without certain fixed parameters
                optimisation_algorithm = RandomizedSearchCV(model, search_parameters, n_jobs=-1,cv=2) #,n_iter=100

            # Fit classifier with parameters returned from search   
            optimised_model = optimisation_algorithm.fit(X_train, y_train)
            for est_val_key in est_vals_keys:
                overall_valls["est_vals"][est_val_key][rep_idx]= optimised_model.best_params_[est_val_key]

        else:

            # Fit moddel with fixed parameters, no optimisation of hyperparameters
            if fixed_parameters:
                optimised_model = model(**fixed_parameters).fit(X_train, y_train.flatten())
            else:
                # Fit moddel without fixed parameters, no optimisation of hyperparameters
                optimised_model = model.fit(X_train, y_train.flatten())

        # Return predicted labels of test set
        y_pred = optimised_model.predict(X_test)


        u_labels = list(set(y_train))

        report = pd.DataFrame(classification_report(y_test,y_pred,output_dict=True))

        cmat = confusion_matrix(y_test,y_pred,labels = u_labels)

        report.loc["accuracy",u_labels] = cmat.diagonal()/cmat.sum(axis=1)

        # Calculate classification metrics
        overall_valls['accuracy'][rep_idx] = np.mean(report.loc['accuracy',:][0:len(u_labels)])
        overall_valls['f1-score'][rep_idx] = np.mean(report.loc['f1-score',:][0:len(u_labels)])
        overall_valls['precision'][rep_idx] = np.mean(report.loc['precision',:][0:len(u_labels)])
        overall_valls['state'][rep_idx] = state

        # Append test and predicted labels
        y_test_a=np.append(y_test_a,y_test)
        y_pred_a=np.append(y_pred_a,y_pred)

        # Append test and predicted nested labels
        if list_nested_label:
            for nested in list_nested_label:
                y_nested_test[nested]=np.append(y_nested_test[nested],np.array(df.loc[index_test,nested]))

        # Append iteration and classifier id
        iteration_id = np.append(iteration_id,[rep_idx]*len(y_pred)) 
        classifier_id = np.append(classifier_id,[model_pack["name"]]*len(y_pred)) 

        # Aggreagate output
    data={
        'y_test_a': y_test_a,
        'y_pred_a': y_pred_a,
        'iteration_id': iteration_id,
        'classifier_id': classifier_id
    }
    data.update(y_nested_test)

    df_ = pd.DataFrame(data=data)

    return df_, overall_valls, len(index_train)


def plot_averaged_scores(all_scores,title,data_folder):
    """Plot average of classification scores
    Args:
        all_scores (pd.DataFrame): Dataframe of classification scores for each run and classifier
        title (str): Addon for file name
        data_folder (str): Path in for saving graphs
    """    
    all_scores = all_scores.rename(columns={"classifier_id":"Classifier"})
    all_data_melt = pd.melt(all_scores,id_vars=['Classifier'],value_vars=['F1 Score','Accuracy'])
    plt.close()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    a = sns.boxplot(x="variable", y="value", hue="Classifier",data=all_data_melt, palette="Set1",ax=ax)
    for line in ax.get_lines()[4::12]:
        line.set_color('yellow')
    for line in ax.get_lines()[10::12]:
        line.set_color('yellow')
    if "call_type" in title:
        ax.set(ylim=(0.35, 1),title="Call type classification",ylabel="Average score")
    else:
        ax.set(ylim=(0.35, 1),title="IVR classification",ylabel="Average score")
    plt.tight_layout()
    a.figure.savefig(f"{title}_average.pdf",format= "pdf")
    a.figure.savefig(f"{title}_average.png")
    plt.close()


def get_aggregated_scores(df_pre_scores,y_test,y_pred,title,data_folder,with_plot):
    """Plot the classification scores of the aggregated labels of the classification runs

    Args:
        df_pre_scores (pd.DataFrame): Dataframe containg all the predicted and test labels (and appending classfier type information)
        y_test (str): String pointing the coloumn in df_pre_scores containing test labels
        y_pred (str): String pointing the coloumn in df_pre_scores containing predicted labels
        title (str): Addon for file name
        data_folder (str): Path in for saving graphs
    """ 
    classifiers = list(df_pre_scores.classifier_id.unique())
    u_labels = list(df_pre_scores[df_pre_scores.classifier_id==classifiers[0]].loc[:,y_test].unique())
    report = pd.DataFrame(
                classification_report(df_pre_scores[df_pre_scores.classifier_id==classifiers[0]].loc[:,y_test].values,
                                    df_pre_scores[df_pre_scores.classifier_id==classifiers[0]].loc[:,y_pred].values,
                                                                    output_dict=True))
    report['Classifier'] = acronym_gen(classifiers[0])

    cmat = confusion_matrix(y_true = df_pre_scores[df_pre_scores.classifier_id==classifiers[0]].loc[:,y_test].values,
                            y_pred = df_pre_scores[df_pre_scores.classifier_id==classifiers[0]].loc[:,y_pred].values,
                            labels = u_labels)

    report.loc["accuracy",u_labels] = cmat.diagonal()/cmat.sum(axis=1)

    if len(classifiers)>1:
        for classifier in classifiers[1:]:
            report_temp = pd.DataFrame(
                classification_report(df_pre_scores[df_pre_scores.classifier_id==classifier].loc[:,y_test].values,
                                    df_pre_scores[df_pre_scores.classifier_id==classifier].loc[:,y_pred].values,
                                                                    output_dict=True))
            report_temp['Classifier'] = acronym_gen(classifier)

            cmat = confusion_matrix(y_true = df_pre_scores[df_pre_scores.classifier_id==classifier].loc[:,y_test].values,
                                    y_pred = df_pre_scores[df_pre_scores.classifier_id==classifier].loc[:,y_pred].values,
                                    labels = u_labels)
            report_temp.loc["accuracy",u_labels] = cmat.diagonal()/cmat.sum(axis=1)
            report = report.append(report_temp)


    
    report['Metric'] = report.index
    report_melt = pd.melt(report,
                        id_vars = ['Classifier','Metric'],
                        value_vars = u_labels)
    report_melt = report_melt.rename(columns={"value":"Value","variable":"Chimp individual"})
    if with_plot:
        plt.close()
        ax = sns.catplot(kind="point",x="Metric",y="Value", hue="Chimp individual", palette="ch:.25", col="Classifier", col_wrap=2,
                        data=report_melt[report_melt["Metric"]!="support"])
        ax.savefig(f"{title}_aggregated_score.pdf",format= "pdf")
        ax.savefig(f"{title}_aggregated_score.png")
        plt.close()

    return report

 
def plot_confusion_matrix_subplot(df,y_test,y_pred,title):
    """ Plot confusion matrix per label type. Axis that incorporates any (optional) nested labels.
        Also creates of subplots (for each classifier) of an classification metric of the users choice (calculated by aggregating over all classification run)

    Args:
        df (pd.DataFrame): Dataframe containg all the predicted and test labels (and appending classfier type information)
        y_test (str): String pointing the coloumn in df containing test labels
        y_pred (str): String pointing the coloumn in df containing predicted labels
        title (str): Addon for file name

    Returns:
        pd.DataFrame: Melted pd.DataFrame of Classfication scores of each classifier implemented
    """    
    

    classifiers = list(df.classifier_id.unique())
    chimps = list(df.y_test_a.unique())
    for classifier in classifiers:
        plt.close()
        with sns.axes_style("white"):
            fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(20,6))
            fig.patch.set_facecolor('white')
            indexx = 0
            cm_m = np.zeros((3, 3))
            for chimp, ax in zip(chimps, axes.flatten()):
                indexx+=1
                target = df[(df.classifier_id==classifier)&(df.y_test_a==chimp)]
                labelz = sorted(list(set(np.append(target.loc[:,y_test].unique(),target.loc[:,y_pred].unique()))),key=str.casefold)
                labelz_0 = sorted(list(set(np.append(target.loc[:,y_test].unique(),target.loc[:,y_pred].unique()))),key=str.casefold)

                cm=confusion_matrix(target.loc[:,y_test].values,target.loc[:,y_pred].values,labels=labelz_0)

                cm = np.asmatrix(pd.DataFrame(cm).iloc[0:3,:])

                for i in range(3):
                    cm_m[i,0] = cm[i,i]
                    cm_m[i,1:3] = cm[i,3:5]

                test = cm_m.transpose()/cm_m.sum(axis=1)
                test = test.transpose()

                sns.heatmap(test.round(3),ax=ax,cmap=plt.cm.Blues,annot=True,fmt='g',square=True, cbar=False)
                ax.title.set_text(chimp)
                ax.set_xticklabels(["Correct"]+labelz[3:5],rotation= 45)
                ax.set_yticklabels(labelz[0:3],rotation = 45)
                if indexx == 1:
                    ax.set_ylabel("Ground truth")
                ax.set_xlabel("Predicted")
                ax.set_facecolor('white')
                plt.tight_layout()
                
        plt.savefig(f"{title}_{classifier}_new.pdf",format="pdf")
        plt.savefig(f"{title}_{classifier}_new.png")
        plt.close()
         

def average_metrics(df_pre_scores,y_test,y_pred):
    """Average classification scores for each classifier type 

    Args:
        df_pre_scores (pd.DataFrame): Dataframe containing the aggregated test and predicted labels of each classifier. Should be indxed by coloums iterationd_id, classifier_id, y_test, y_pred
        y_test (str): Name of the column of containing the test labels
        y_pred (str):  Name of the column containing the predicted labels

    Returns:
        [pd.DataFrame]: Dataframe with averaged classification f1, recall and accuracy scores
    """    
    f1_scores = df_pre_scores.groupby(["iteration_id","classifier_id"]).apply(lambda x: f1_score(x[y_test],x[y_pred],average="macro")).to_frame().rename(columns={0: "F1 Score"})
    recall = df_pre_scores.groupby(["iteration_id","classifier_id"]).apply(lambda x: recall_score(x[y_test],x[y_pred],average="macro")).to_frame().rename(columns={0: "Recall"})
    accuracy = df_pre_scores.groupby(["iteration_id","classifier_id"]).apply(lambda x: accuracy_score(x[y_test],x[y_pred])).to_frame().rename(columns={0: "Accuracy"})
    f1_scores.reset_index(inplace=True,drop=False)
    recall.reset_index(inplace=True,drop=True,)
    accuracy.reset_index(inplace=True,drop=True)
    all_scores = pd.concat([f1_scores,recall,accuracy],axis=1)
    return all_scores
