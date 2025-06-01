import numpy as np
import time
from IPython.display import clear_output
from copy import copy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class RBM:
    def __init__(self, visible_size, hidden_size, learning_rate=0.01, momentum=0.5, weight_decay=0.0001, k=1):
        """
        Inicializa una Restricted Boltzmann Machine
        
        Parámetros:
        - visible_size: número de unidades visibles (dimensión de entrada)
        - hidden_size: número de unidades ocultas
        - learning_rate: tasa de aprendizaje para actualizar los pesos
        - momentum: factor de momentum para evitar mínimos locales
        - weight_decay: regularización para evitar sobreajuste
        - k: número de pasos de muestreo en Contrastive Divergence (CD-k)
        """
        # Inicialización de pesos y sesgos
        self.weights = np.random.normal(0, 0.01, (visible_size, hidden_size))
        self.visible_bias = np.zeros(visible_size)
        self.hidden_bias = np.zeros(hidden_size)
        
        # Hiperparámetros
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.k = k
        
        # Incrementos anteriores para momentum
        self.prev_weight_inc = np.zeros((visible_size, hidden_size))
        self.prev_visible_bias_inc = np.zeros(visible_size)
        self.prev_hidden_bias_inc = np.zeros(hidden_size)
        
        # Almacenar el error para seguimiento
        self.reconstruction_errors = []
    
    def sample_hidden_given_visible(self, visible_states):
        """
        Muestrea estados ocultos dadas las activaciones visibles
        
        Parámetros:
        - visible_states: estados de la capa visible
        
        Retorna:
        - hidden_probs: probabilidades de activación de la capa oculta
        - hidden_states: estados muestreados de la capa oculta
        """
        # Calcular probabilidades de activación: P(h=1|v)
        hidden_activations = np.dot(visible_states, self.weights) + self.hidden_bias
        hidden_probs = sigmoid(hidden_activations)
        
        # Muestrear estados binarios
        random_matrix = np.random.random(hidden_probs.shape)
        hidden_states = (hidden_probs > random_matrix).astype(np.float32)
        
        return hidden_probs, hidden_states
    
    def sample_visible_given_hidden(self, hidden_states):
        """
        Muestrea estados visibles dados los estados ocultos
        
        Parámetros:
        - hidden_states: estados de la capa oculta
        
        Retorna:
        - visible_probs: probabilidades de activación de la capa visible
        - visible_states: estados muestreados de la capa visible
        """
        # Calcular probabilidades de activación: P(v=1|h)
        visible_activations = np.dot(hidden_states, self.weights.T) + self.visible_bias
        visible_probs = sigmoid(visible_activations)
        
        # Muestrear estados binarios
        random_matrix = np.random.random(visible_probs.shape)
        visible_states = (visible_probs > random_matrix).astype(np.float32)
        
        return visible_probs, visible_states
    
    def contrastive_divergence(self, batch_data, verbose=False):
        """
        Implementación del algoritmo de Contrastive Divergence (CD-k)
        
        Parámetros:
        - batch_data: lote de datos de entrada (batch)
        - verbose: si es True, muestra información del progreso
        
        Retorna:
        - error: error de reconstrucción promedio en este batch
        """
        # Paso 1: Inicializar estados visibles con datos de entrada
        v0 = batch_data
        
        # Paso 2: Calcular probabilidades y estados de las unidades ocultas dadas las visibles
        h0_probs, h0 = self.sample_hidden_given_visible(v0)
        
        # Muestras iniciales para las estadísticas positivas
        positive_associations = np.dot(v0.T, h0_probs)
        
        # Inicialización para la fase negativa
        if not isinstance(v0, np.ndarray):
            v0 = v0.numpy()

        vk = v0.copy()
        hk = h0.copy()
        
        # Paso 3: Realizar k pasos de muestreo de Gibbs (CD-k)
        for i in range(self.k):
            # Muestrear nuevos estados visibles a partir de los estados ocultos actuales
            vk_probs, vk = self.sample_visible_given_hidden(hk)
            
            # Muestrear nuevos estados ocultos a partir de los estados visibles recién generados
            hk_probs, hk = self.sample_hidden_given_visible(vk)
            
            if verbose and i == 0:
                print(f"Muestreo de Gibbs - Paso {i+1}")
                print(f"Probabilidades ocultas: min={hk_probs.min():.4f}, max={hk_probs.max():.4f}")
                print(f"Probabilidades visibles: min={vk_probs.min():.4f}, max={vk_probs.max():.4f}")
        
        # Para el último paso, usamos las probabilidades en lugar de los estados muestreados
        hk_probs, _ = self.sample_hidden_given_visible(vk)
        
        # Paso 4: Calcular estadísticas negativas
        negative_associations = np.dot(vk.T, hk_probs)
        
        # Paso 5: Calcular gradientes y actualizar parámetros con momentum
        
        # Gradiente para los pesos: <v_i * h_j>_data - <v_i * h_j>_model
        weight_gradient = (positive_associations - negative_associations) / batch_data.shape[0]
        
        # Gradiente para los sesgos
        visible_bias_gradient = np.mean(v0 - vk, axis=0)
        hidden_bias_gradient = np.mean(h0_probs - hk_probs, axis=0)
        
        # Actualización con momentum y weight decay
        weight_inc = (self.learning_rate * weight_gradient - 
                     self.weight_decay * self.weights + 
                     self.momentum * self.prev_weight_inc)
        
        visible_bias_inc = self.learning_rate * visible_bias_gradient + self.momentum * self.prev_visible_bias_inc
        hidden_bias_inc = self.learning_rate * hidden_bias_gradient + self.momentum * self.prev_hidden_bias_inc
        
        # Actualizar parámetros
        self.weights += weight_inc
        self.visible_bias += visible_bias_inc
        self.hidden_bias += hidden_bias_inc
        
        # Guardar los incrementos para el próximo uso del momentum
        self.prev_weight_inc = weight_inc
        self.prev_visible_bias_inc = visible_bias_inc
        self.prev_hidden_bias_inc = hidden_bias_inc
        
        # Calcular error de reconstrucción
        reconstruction_error = np.mean((v0 - vk) ** 2)
        self.reconstruction_errors.append(reconstruction_error)
        
        if verbose:
            print(f"Error de reconstrucción: {reconstruction_error:.6f}")
        
        return reconstruction_error
    
    def train(self, data, epochs=10, batch_size=100):
        """
        Entrenamiento de la RBM usando Contrastive Divergence
        
        Parámetros:
        - data: datos de entrenamiento
        - epochs: número de épocas de entrenamiento
        - batch_size: tamaño del lote (batch)
        """
        n_samples = data.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            epoch_error = 0
            start_time = time.time()
            
            # Mezclar los datos en cada época
            indices = np.random.permutation(n_samples)
            data_shuffled = data[indices]
            
            for batch in range(n_batches):
                batch_start = batch * batch_size
                batch_end = min((batch + 1) * batch_size, n_samples)
                batch_data = data_shuffled[batch_start:batch_end]
                batch_data = batch_data.reshape(batch_data.shape[0], -1)
                
                # Aplicar Contrastive Divergence
                batch_error = self.contrastive_divergence(batch_data, verbose=(batch==0 and epoch==0))
                epoch_error += batch_error
            
            # Error promedio en la época
            avg_epoch_error = epoch_error / n_batches
            elapsed_time = time.time() - start_time
            
            print(f"Época {epoch+1}/{epochs}, Error: {avg_epoch_error:.6f}, Tiempo: {elapsed_time:.2f}s")
            
    
    def get_hidden_features(self, data):
        """
        Obtener características de la capa oculta para los datos
        
        Parámetros:
        - data: datos de entrada
        
        Retorna:
        - hidden_features: características extraídas por la capa oculta
        """
        hidden_probs, _ = self.sample_hidden_given_visible(data)
        return hidden_probs

# -------------------------------------------------------------------------------
# Clase para Deep Belief Network (DBN)
# -------------------------------------------------------------------------------
class DBN:
    def __init__(self, layer_sizes, learning_rates=None, momentum=0.5, weight_decay=0.0001, k=1):
        """
        Inicializa una Deep Belief Network compuesta por múltiples RBMs
        
        Parámetros:
        - layer_sizes: lista con el tamaño de cada capa (incluyendo la visible)
        - learning_rates: tasas de aprendizaje para cada RBM
        - momentum: factor de momentum
        - weight_decay: regularización
        - k: pasos de Contrastive Divergence
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # Número de RBMs
        
        # Configurar tasas de aprendizaje
        if learning_rates is None:
            self.learning_rates = [0.01] * self.n_layers
        else:
            self.learning_rates = learning_rates
        
        # Inicializar RBMs
        self.rbms = []
        for i in range(self.n_layers):
            self.rbms.append(
                RBM(
                    visible_size=layer_sizes[i],
                    hidden_size=layer_sizes[i+1],
                    learning_rate=self.learning_rates[i],
                    momentum=momentum,
                    weight_decay=weight_decay,
                    k=k
                )
            )
    
    def pretrain(self, data, epochs_per_layer=10, batch_size=100):
        """
        Entrenamiento no supervisado capa por capa (greedy layer-wise)
        
        Parámetros:
        - data: datos de entrenamiento
        - epochs_per_layer: épocas para cada capa
        - batch_size: tamaño del lote
        """
        print("Iniciando el pre-entrenamiento de la DBN")
        
        # Entrenar cada RBM de manera secuencial
        input_data = copy(data)
        input_data = input_data.numpy()
        
        for i in range(self.n_layers):
            print(f"\n=== Entrenando RBM {i+1}/{self.n_layers} ===")
            print(f"Capa {i+1}: {self.layer_sizes[i]} -> {self.layer_sizes[i+1]}")
            
            # Entrenar la RBM actual
            self.rbms[i].train(
                input_data, 
                epochs=epochs_per_layer, 
                batch_size=batch_size, 
            )
            
            # Los datos de entrada para la siguiente capa son las activaciones ocultas
            input_data = self.rbms[i].get_hidden_features(input_data)
            
            print(f"Forma de la salida de la capa {i+1}: {input_data.shape}")
        
        print("\n¡Pre-entrenamiento completado!")
    
    def forward_pass(self, data):
        """
        Propagación hacia adelante a través de todas las capas
        
        Parámetros:
        - data: datos de entrada
        
        Retorna:
        - hidden_features: características de la última capa oculta
        """
        input_data = data.clone()
        
        for i in range(self.n_layers):
            input_data = self.rbms[i].get_hidden_features(input_data)
        
        return input_data
    
    def reconstruct(self, data, layer=-1):
        """
        Reconstruye los datos de entrada propagando hacia adelante y luego hacia atrás
        
        Parámetros:
        - data: datos de entrada
        - layer: hasta qué capa propagar (-1 significa hasta la última)
        
        Retorna:
        - reconstruction: reconstrucción de los datos de entrada
        """
        if layer == -1:
            layer = self.n_layers - 1
        
        # Propagación hacia adelante
        forward_data = data.clone()
        for i in range(layer + 1):
            h_probs, h_states = self.rbms[i].sample_hidden_given_visible(forward_data)
            forward_data = h_probs
        
        # Propagación hacia atrás
        backward_data = forward_data.copy()
        for i in range(layer, -1, -1):
            v_probs, v_states = self.rbms[i].sample_visible_given_hidden(backward_data)
            backward_data = v_probs
        
        return backward_data
    

    def get_features(self, data):
        """
        Extrae características usando todas las capas entrenadas
        
        Parámetros:
        - data: datos de entrada
        
        Retorna:
        - features: características extraídas
        """
        return self.forward_pass(data)