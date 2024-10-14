import numpy as np
import util
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import torch


class LogitProbEstimator:
    @staticmethod
    def evaluate(preds, inputs, masks):
        all_f1_scores = []
        
        for pred, input_i, mask in tqdm(zip(preds, inputs, masks), total=len(preds), desc="Processing samples"):
            f1_scores = LogitProbEstimator.process_sample(pred, input_i, mask)
            all_f1_scores.append(f1_scores)
    
        return all_f1_scores

    @staticmethod
    def process_sample(pred, input_i, mask):
        preds_train, preds_test = pred.value
        labels_train, labels_test = mask.value

        name_of_chart = pred.model, pred.spectrum, pred.fold
        
        logists_train  = transform_to_two_channel(preds_train, labels_train)
        logits_test  = transform_to_two_channel(preds_test, labels_test)
        
        #histogram(logists_train, logits_test, name_of_chart)
        #final_classes_images = apply_gmm(preds_test, logists_train, n_components=3)
        probs = apply_kde(preds_test, logists_train)        
        roc_soft, roc_prob = evaluate_performance_likehood(preds_test, probs, labels_test)
        
        return roc_soft, roc_prob
        
def transform_to_two_channel(logits, labels):
    logits_class_1, logits_class_0 = [], []
    
    for n_logits, n_labels in zip(logits, labels):
        n_logits = n_logits[0]
        n_labels = n_labels[0]
        
        # Extract index with label 1 and 0
        index_1 = np.argwhere(n_labels == 1)
        index_0 = np.argwhere(n_labels == 0)
        
        # Extract logits using index from label 1 and 0
        n_logits_class_1 = n_logits[tuple(index_1.T)]
        n_logits_class_0 = n_logits[tuple(index_0.T)]
        
        logits_class_1.extend(n_logits_class_1)
        logits_class_0.extend(n_logits_class_0) 
    
    return np.array(logits_class_1), np.array(logits_class_0)


def apply_kde(logits_test, logits_train):
   
    def gaussian_kernel(u):
        return (1 / math.sqrt(2 * math.pi)) * np.exp(-0.5 * u**2)

    # Função para calcular o KDE
    def kde(x, data):
        n = len(data)
        sigma = np.std(data)
        h = (4 * sigma**5 / (3 * n))**(1/5)

        result = 0
        for xi in data:
            result += gaussian_kernel((x - xi) / h)
        return result / (n * h)

    logits_pos_train, logits_neg_train = logits_train

    likelyhood = kde_evaluate(kde, logits_pos_train, logits_neg_train, logits_test)

    # Reshape para a forma original das imagens de teste
    likelyhood_matrix = likelyhood.reshape(logits_test.shape[0], logits_test.shape[1], logits_test.shape[2], logits_test.shape[3])

    return likelyhood_matrix

def kde_evaluate(kde, data_pos_train, data_neg_train, data_test):

    add_smooth = 0.000001

    sample_pos = calculate_sample(data_pos_train)
    sample_neg = calculate_sample(data_neg_train)

    # Definir o mesmo intervalo de x para ambas as distribuições
    min_value = math.ceil(min(sample_pos.min(), sample_neg.min()))
    max_value = math.ceil(max(sample_pos.max(), sample_neg.max()))
    
    # Valores de x comuns para ambas as distribuições
    x_values = np.linspace(min_value, max_value, 15)  # Escolha de N pontos

    # Calcular as densidades positivas e negativas usando o KDE
    dens_values_pos = [kde(x, sample_pos) for x in x_values]
    dens_values_neg = [kde(x, sample_neg) for x in x_values]

    data_test = data_test.flatten()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Definir os valores de likelihood para positiva e negativa
    likelyhood_pos = linear_interpolation(x_values, dens_values_pos, data_test, add_smooth, device)
    likelyhood_neg = linear_interpolation(x_values, dens_values_neg, data_test, add_smooth, device)

    # Fazer o cálculo final de likelihood na GPU
    likelyhood = (likelyhood_pos + add_smooth) / (likelyhood_pos + likelyhood_neg + 2 * add_smooth)

    # Trazer os dados de volta para o numpy se necessário
    likelyhood = likelyhood.cpu().numpy()

    # Plotar o gráfico com ambas as likelihoods
    plot_likelihood(x_values, dens_values_pos, dens_values_neg)

    return likelyhood

def kde_cumulative(kde, data_pos_train, data_neg_train, data_test):

    add_smooth = 0.000001

    sample_pos = calculate_sample(data_pos_train)
    sample_neg = calculate_sample(data_neg_train)

    # Definir o mesmo intervalo de x para ambas as distribuições
    min_value = math.ceil(min(sample_pos.min(), sample_neg.min()))
    max_value = math.ceil(max(sample_pos.max(), sample_neg.max()))
    
    # Valores de x comuns para ambas as distribuições
    x_values = np.linspace(min_value, max_value, 15)  # Escolha de N pontos

    # Calcular as densidades positivas e negativas usando o KDE
    dens_values_pos = [kde(x, sample_pos) for x in x_values]
    dens_values_neg = [kde(x, sample_neg) for x in x_values]

    # Calcular a soma acumulada das densidades (KDE acumulado)
    cdf_values_pos = np.cumsum(dens_values_pos)
    cdf_values_neg = np.cumsum(dens_values_neg)

    # Normalizar o CDF para que o último valor seja 1
    cdf_values_pos /= cdf_values_pos[-1]
    cdf_values_neg /= cdf_values_neg[-1]

    data_test = data_test.flatten()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Definir os valores de likelihood para positiva e negativa usando a versão acumulada (CDF)
    likelyhood_pos = linear_interpolation(x_values, cdf_values_pos, data_test, add_smooth, device)
    likelyhood_neg = linear_interpolation(x_values, cdf_values_neg, data_test, add_smooth, device)

    # Fazer o cálculo final de likelihood na GPU
    #likelyhood = (likelyhood_pos + add_smooth) / (likelyhood_pos + likelyhood_neg + 2 * add_smooth)
    likelyhood = likelyhood_pos

    # Trazer os dados de volta para o numpy se necessário
    likelyhood = likelyhood.cpu().numpy()
  
    # Plotar o gráfico com ambas as likelihoods
   # plot_likelihood(x_values, cdf_values_pos, cdf_values_neg)

    return likelyhood


def plot_likelihood(x_values, dens_values_pos, dens_values_neg):
    # Criar o gráfico com as duas curvas de densidade (positiva e negativa)
    plt.plot(x_values, dens_values_pos, label='KDE Positive Train', color='blue')
    plt.plot(x_values, dens_values_neg, label='KDE Negative Train', color='orange')
    plt.title('Kernel Density Estimation - Likelihoods')
    plt.xlabel('logits')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"post_training/histograms/kde_chart.png")

def calculate_sample(data):
    N = len(data)
    Z = 1.96
    p = 0.5
    E = 0.05

    n = (Z**2 * p * (1 - p)) / (E**2)
    n_adjusted = n / (1 + (n - 1) / N)
    n_sample = math.ceil(n_adjusted)

    return np.random.choice(data, size=n_sample, replace=False) 

def linear_interpolation(x_values, y_values, x_test, add_smooth, device):
    # Converta para tensores e envie para a GPU
    x_values = torch.tensor(x_values, device=device, dtype=torch.float32)
    y_values = torch.tensor(y_values, device=device, dtype=torch.float32)
    x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
    

    # Para cada valor em x_test, encontrar os dois índices em x_values que ele fica entre
    indices = torch.searchsorted(x_values, x_test)

    # Corrigir os índices para lidar com limites
    indices = torch.clamp(indices, 1, len(x_values) - 1)

    # Obtenha os dois pontos de x e y que cercam cada ponto de x_test
    x0 = x_values[indices - 1]
    x1 = x_values[indices]
    y0 = y_values[indices - 1]
    y1 = y_values[indices]

    # Interpolação linear vetorizada
    weight = (x_test - x0) / (x1 - x0)
    interpolated_values = y0 + weight * (y1 - y0)

    # Aplicar a suavização de Laplace
    interpolated_values = interpolated_values + add_smooth

    return interpolated_values

def evaluate_performance_likehood(logits_test, probs, labels_test):
    logits_test = util.sigmoid(logits_test)
   # logits_test = util.thresholding(logits_test)
    #f1s_soft, _ = util.perfomance_metrics(logits_test, labels_test)
    #roc_auc_soft = util.calculate_roc_auc(logits_test, labels_test)
    #prec_auc_soft = util.calculate_precision_recall_ap(logits_test, labels_test)
    ece_soft = calculate_ece(logits_test, labels_test)
    
    #probs = util.thresholding(probs)
    #f1s_prob, _ = util.perfomance_metrics(probs, labels_test)
    #roc_auc_prob = util.calculate_roc_auc(probs, labels_test)
    prec_auc_prob = util.calculate_precision_recall_ap(probs, labels_test)
    ece_prob = calculate_ece(probs, labels_test)

    return round(ece_soft*100, 2), round(ece_prob*100, 2)
   
def histogram(logits_train, logits_test, name_of_chart):
    train_pos, train_neg = logits_train
    test_pos, test_neg = logits_test
    model, spec, fold = name_of_chart

    train_pos_flat = train_pos.flatten()
    train_neg_flat = train_neg.flatten()

    test_pos_flat = test_pos.flatten()
    test_neg_flat = test_neg.flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    plt.suptitle(f'Charts for {model.upper()} - {spec.upper()} - FOLD {fold}', fontsize=16)
    
    ax1.hist(train_pos_flat, bins=30, histtype='step', linewidth=1.5, label='Positive Train Logits', density=True)
    ax1.hist(train_neg_flat, bins=30, histtype='step', linewidth=1.5, label='Negative Train Logits', density=True)
    ax1.set_title('Histogram of Train Logits')
    ax1.set_xlabel('Logit')
    ax1.set_ylabel('Density')
    ax1.legend(loc='upper right')

    ax2.hist(test_pos_flat, bins=30, histtype='step', linewidth=1.5, label='Positive Test Logits', density=True)
    ax2.hist(test_neg_flat, bins=30, histtype='step', linewidth=1.5, label='Negative Test Logits', density=True)
    ax2.set_title('Histogram of Test Logits')
    ax2.set_xlabel('Logit')
    ax2.set_ylabel('Density')
    ax2.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    file_name = f'densityxlogits_histogram_{model}_{spec}_fold{fold}.png'
    plt.savefig(f"post_training/histograms/{file_name}")
    plt.close()

def calculate_ece(probs, labels, n_bins=10):
    # Inicializa os bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    total_samples = len(labels)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Seleciona as previsões que caem no bin atual
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Calcula a confiança média no bin
            avg_confidence_in_bin = np.mean(probs[in_bin])
            # Calcula a precisão no bin
            avg_accuracy_in_bin = np.mean(labels[in_bin] == (probs[in_bin] > 0.5))
            # Adiciona ao ECE ponderado pelo número de exemplos no bin
            ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin

    return ece


""" def apply_gmm(logits_test, logits_train, n_components=2):
    
    logits_pos_train, logits_neg_train = logits_train

    logits_pos_train_flat = logits_pos_train.reshape(-1, 1)
    logits_neg_train_flat = logits_neg_train.reshape(-1, 1)
    logits_test_flat = logits_test.reshape(-1, 1)

    # Ajustar GMM para os dados de treino
    gmm_pos = GaussianMixture(n_components=n_components, random_state=0)
    gmm_pos.fit(logits_pos_train_flat)
    
    # Ajustar GMM para os dados de treino
    gmm_neg = GaussianMixture(n_components=n_components, random_state=0)
    gmm_neg.fit(logits_neg_train_flat)

    # Avaliar as likelihoods para os dados de teste
    likelihood_pos = gmm_pos.score_samples(logits_test_flat)
    likelihood_neg = gmm_neg.score_samples(logits_test_flat)

    # Normalize the likelihoods
    likelihood_pos_norm = (likelihood_pos - likelihood_pos.min()) / (likelihood_pos.max() - likelihood_pos.min())
    likelihood_neg_norm = (likelihood_neg - likelihood_neg.min()) / (likelihood_neg.max() - likelihood_neg.min())
    
    final_classes = np.maximum(likelihood_pos_norm, likelihood_neg_norm)
    
    # Reshapeando o resultado para a forma original das imagens de teste 
    final_classes_images = final_classes.reshape(logits_test.shape[0], logits_test.shape[1], logits_test.shape[2], logits_test.shape[3])
    
    return final_classes_images """


""" 
    def kde_evaluate(kde, data_pos_train, data_neg_train, data_test):
    
        ....

        interp_pos = interp1d(x_values_pos, dens_values_pos, bounds_error=False, fill_value=add_smooth)
        interp_neg = interp1d(x_values_neg, dens_values_neg, bounds_error=False, fill_value=add_smooth)
        
        likelyhood_pos = [interp_pos(x) for x in tqdm(data_test, desc="Interpolating Pos Density")]
        likelyhood_neg = [interp_neg(x) for x in tqdm(data_test, desc="Interpolating Neg Density")]

        likelyhood_pos = np.array(likelyhood_pos)
        likelyhood_neg = np.array(likelyhood_neg)

        likelyhood = (likelyhood_pos + add_smooth) / (likelyhood_pos + likelyhood_neg + 2 * add_smooth) """




