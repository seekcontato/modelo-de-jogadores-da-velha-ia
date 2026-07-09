import numpy as np
import random
import json
import os
from datetime import datetime

# =============================================================================
# CONFIGURACOES
# =============================================================================

POPULACAO_TAMANHO = 200       # Numero de redes neurais por jogador (X e O)
GERACOES = 1000              # Quantas geracoes de evolucao
PARTIDAS_POR_AVALIACAO = 30  # Partidas que cada rede joga para ser avaliada
TAXA_MUTACAO = 0.15          # Chance de mutar um peso
INTENSIDADE_MUTACAO = 0.5    # Quanto um peso pode mudar
TOP_ELITE = 5                # Melhores redes que sobrevivem intactas
CAMADAS_OCULTAS = [64, 32]   # Arquitetura da rede neural

# =============================================================================
# REDE NEURAL
# =============================================================================

class RedeNeural:
    def __init__(self, tamanho_entrada=9, tamanho_saida=9, camadas_ocultas=None):
        if camadas_ocultas is None:
            camadas_ocultas = CAMADAS_OCULTAS
        
        self.tamanho_entrada = tamanho_entrada
        self.tamanho_saida = tamanho_saida
        self.camadas = [tamanho_entrada] + list(camadas_ocultas) + [tamanho_saida]
        self.pesos = []
        self.biases = []
        self.fitness = 0
        self.vitorias = 0
        self.derrotas = 0
        self.empates = 0
        self.geracao = 0
        
        # Inicializa pesos com Xavier/Glorot
        for i in range(len(self.camadas) - 1):
            limite = np.sqrt(6.0 / (self.camadas[i] + self.camadas[i+1]))
            w = np.random.uniform(-limite, limite, (self.camadas[i], self.camadas[i+1]))
            b = np.random.uniform(-limite, limite, (self.camadas[i+1],))
            self.pesos.append(w)
            self.biases.append(b)
    
    def ativacao(self, x):
        """ReLU"""
        return np.maximum(0, x)
    
    def softmax(self, x):
        """Softmax para saida probabilistica"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def prever(self, estado_tabuleiro):
        """
        Recebe o tabuleiro (lista de 9 posicoes: 0=vazio, 1=X, -1=O)
        Retorna probabilidades para cada posicao (0-8)
        """
        x = np.array(estado_tabuleiro, dtype=np.float32)
        
        for i in range(len(self.pesos) - 1):
            x = self.ativacao(np.dot(x, self.pesos[i]) + self.biases[i])
        
        # Ultima camada com softmax
        x = np.dot(x, self.pesos[-1]) + self.biases[-1]
        return self.softmax(x)
    
    def escolher_movimento(self, estado_tabuleiro, jogador_atual):
        """
        Escolhe o melhor movimento valido baseado nas probabilidades da rede
        """
        probs = self.prever(estado_tabuleiro)
        
        # Cria mascara de movimentos validos (posicoes vazias = 0 no tabuleiro)
        validos = [i for i in range(9) if estado_tabuleiro[i] == 0]
        
        if not validos:
            return None
        
        # Filtra apenas probabilidades de movimentos validos
        probs_validas = [(i, probs[i]) for i in validos]
        probs_validas.sort(key=lambda x: x[1], reverse=True)
        
        # Escolhe o melhor movimento valido
        return probs_validas[0][0]
    
    def copiar(self):
        """Cria uma copia profunda da rede"""
        nova = RedeNeural(self.tamanho_entrada, self.tamanho_saida, CAMADAS_OCULTAS)
        nova.pesos = [w.copy() for w in self.pesos]
        nova.biases = [b.copy() for b in self.biases]
        nova.fitness = self.fitness
        nova.geracao = self.geracao
        return nova
    
    def mutar(self, taxa=TAXA_MUTACAO, intensidade=INTENSIDADE_MUTACAO):
        """Aplica mutacao nos pesos e biases"""
        for i in range(len(self.pesos)):
            mascara = np.random.random(self.pesos[i].shape) < taxa
            self.pesos[i] += mascara * np.random.normal(0, intensidade, self.pesos[i].shape)
            
            mascara_b = np.random.random(self.biases[i].shape) < taxa
            self.biases[i] += mascara_b * np.random.normal(0, intensidade, self.biases[i].shape)
    
    def cruzar(self, outra):
        """Crossover entre duas redes (media ponderada pelo fitness)"""
        filho = self.copiar()
        total_fitness = self.fitness + outra.fitness + 1e-8
        peso_self = self.fitness / total_fitness
        peso_outra = outra.fitness / total_fitness
        
        for i in range(len(self.pesos)):
            filho.pesos[i] = peso_self * self.pesos[i] + peso_outra * outra.pesos[i]
            filho.biases[i] = peso_self * self.biases[i] + peso_outra * outra.biases[i]
        
        return filho
    
    def salvar(self, caminho):
        """Salva a rede em arquivo JSON"""
        dados = {
            'camadas': self.camadas,
            'pesos': [w.tolist() for w in self.pesos],
            'biases': [b.tolist() for b in self.biases],
            'fitness': float(self.fitness),
            'vitorias': self.vitorias,
            'derrotas': self.derrotas,
            'empates': self.empates,
            'geracao': self.geracao
        }
        with open(caminho, 'w') as f:
            json.dump(dados, f)
    
    @staticmethod
    def carregar(caminho):
        """Carrega uma rede de arquivo JSON"""
        with open(caminho, 'r') as f:
            dados = json.load(f)
        
        rn = RedeNeural(dados['camadas'][0], dados['camadas'][-1], dados['camadas'][1:-1])
        rn.pesos = [np.array(w) for w in dados['pesos']]
        rn.biases = [np.array(b) for b in dados['biases']]
        rn.fitness = dados['fitness']
        rn.vitorias = dados['vitorias']
        rn.derrotas = dados['derrotas']
        rn.empates = dados['empates']
        rn.geracao = dados['geracao']
        return rn


# =============================================================================
# JOGO DA VELHA
# =============================================================================

class JogoVelha:
    def __init__(self):
        self.tabuleiro = [0] * 9  # 0=vazio, 1=X, -1=O
        self.historico = []
    
    def reset(self):
        self.tabuleiro = [0] * 9
        self.historico = []
    
    def jogar(self, posicao, jogador):
        """Jogador: 1 = X, -1 = O"""
        if self.tabuleiro[posicao] == 0:
            self.tabuleiro[posicao] = jogador
            self.historico.append((self.tabuleiro.copy(), posicao, jogador))
            return True
        return False
    
    def verificar_vencedor(self):
        """Retorna 1 (X vence), -1 (O vence), 0 (empate), None (continua)"""
        combinacoes = [
            [0,1,2], [3,4,5], [6,7,8],  # linhas
            [0,3,6], [1,4,7], [2,5,8],  # colunas
            [0,4,8], [2,4,6]             # diagonais
        ]
        
        for c in combinacoes:
            if self.tabuleiro[c[0]] == self.tabuleiro[c[1]] == self.tabuleiro[c[2]] != 0:
                return self.tabuleiro[c[0]]
        
        if 0 not in self.tabuleiro:
            return 0  # Empate
        
        return None
    
    def movimentos_validos(self):
        return [i for i in range(9) if self.tabuleiro[i] == 0]
    
    def __str__(self):
        simbolos = {0: ' ', 1: 'X', -1: 'O'}
        linhas = []
        for i in range(3):
            linha = ' | '.join(simbolos[self.tabuleiro[i*3 + j]] for j in range(3))
            linhas.append(f' {linha} ')
        return '\n' + '\n-----------\n'.join(linhas) + '\n'


# =============================================================================
# TORNEIO E AVALIACAO
# =============================================================================

def jogar_partida(rede_x, rede_o, verbose=False):
    """
    Faz duas redes neurais jogarem uma partida.
    Retorna: 1 se X vence, -1 se O vence, 0 se empate
    """
    jogo = JogoVelha()
    jogador_atual = 1  # X comeca
    
    while True:
        if jogador_atual == 1:
            movimento = rede_x.escolher_movimento(jogo.tabuleiro, 1)
        else:
            movimento = rede_o.escolher_movimento(jogo.tabuleiro, -1)
        
        if movimento is None:
            break
        
        jogo.jogar(movimento, jogador_atual)
        
        if verbose:
            print(jogo)
        
        resultado = jogo.verificar_vencedor()
        if resultado is not None:
            return resultado, jogo.historico
        
        jogador_atual *= -1
    
    return 0, jogo.historico


def avaliar_rede(rede, populacao_oponente, num_partidas=PARTIDAS_POR_AVALIACAO, jogador=1):
    """
    Avalia uma rede neural jogando contra varios oponentes aleatorios da populacao adversaria.
    Retorna o fitness (pontuacao acumulada).
    """
    fitness = 0
    vitorias = 0
    derrotas = 0
    empates = 0
    
    oponentes = random.sample(populacao_oponente, min(num_partidas, len(populacao_oponente)))
    
    for oponente in oponentes:
        if jogador == 1:
            resultado, _ = jogar_partida(rede, oponente)
        else:
            resultado, _ = jogar_partida(oponente, rede)
        
        if resultado == jogador:
            fitness += 3
            vitorias += 1
        elif resultado == 0:
            fitness += 1
            empates += 1
        else:
            fitness -= 2
            derrotas += 1
    
    rede.fitness = fitness
    rede.vitorias = vitorias
    rede.derrotas = derrotas
    rede.empates = empates
    
    return fitness


def proxima_geracao(populacao, top_elite=TOP_ELITE):
    """
    Cria a proxima geracao usando selecao por torneio + crossover + mutacao
    """
    # Ordena por fitness (maior primeiro)
    populacao.sort(key=lambda r: r.fitness, reverse=True)
    
    nova_populacao = []
    
    # Elitismo: mantem os melhores
    for i in range(top_elite):
        elite = populacao[i].copiar()
        elite.geracao = populacao[i].geracao + 1
        nova_populacao.append(elite)
    
    # Preenche o resto com crossover e mutacao
    while len(nova_populacao) < len(populacao):
        # Selecao por torneio
        pai1 = max(random.sample(populacao[:len(populacao)//2], 3), key=lambda r: r.fitness)
        pai2 = max(random.sample(populacao[:len(populacao)//2], 3), key=lambda r: r.fitness)
        
        filho = pai1.cruzar(pai2)
        filho.mutar()
        filho.geracao = pai1.geracao + 1
        nova_populacao.append(filho)
    
    return nova_populacao


# =============================================================================
# TREINAMENTO PRINCIPAL
# =============================================================================

def treinar(verbose=True, salvar_cada=20):
    """
    Treina duas populacoes de redes neurais (X e O) competindo entre si
    """
    print("=" * 70)
    print("  JOGO DA VELHA - REDES NEURAIS COMPETITIVAS")
    print("=" * 70)
    print(f"\nConfiguracoes:")
    print(f"  Populacao: {POPULACAO_TAMANHO} redes por jogador")
    print(f"  Geracoes: {GERACOES}")
    print(f"  Partidas/avaliacao: {PARTIDAS_POR_AVALIACAO}")
    print(f"  Taxa de mutacao: {TAXA_MUTACAO}")
    print(f"  Arquitetura: 9 -> {' -> '.join(map(str, CAMADAS_OCULTAS))} -> 9")
    print()
    
    # Cria populacoes iniciais
    populacao_x = [RedeNeural() for _ in range(POPULACAO_TAMANHO)]
    populacao_o = [RedeNeural() for _ in range(POPULACAO_TAMANHO)]
    
    melhor_x = None
    melhor_o = None
    
    for geracao in range(1, GERACOES + 1):
        # Avalia populacao X contra populacao O
        for i, rede in enumerate(populacao_x):
            avaliar_rede(rede, populacao_o, PARTIDAS_POR_AVALIACAO, jogador=1)
        
        # Avalia populacao O contra populacao X
        for i, rede in enumerate(populacao_o):
            avaliar_rede(rede, populacao_x, PARTIDAS_POR_AVALIACAO, jogador=-1)
        
        # Estatisticas
        populacao_x.sort(key=lambda r: r.fitness, reverse=True)
        populacao_o.sort(key=lambda r: r.fitness, reverse=True)
        
        melhor_x = populacao_x[0]
        melhor_o = populacao_o[0]
        
        if verbose and geracao % 5 == 0:
            print(f"\n{'-' * 70}")
            print(f"  GERACAO {geracao}/{GERACOES}")
            print(f"{'-' * 70}")
            print(f"  Melhor X  -> Fitness: {melhor_x.fitness:6.1f} | "
                  f"V: {melhor_x.vitorias:2d} | D: {melhor_x.derrotas:2d} | E: {melhor_x.empates:2d}")
            print(f"  Melhor O  -> Fitness: {melhor_o.fitness:6.1f} | "
                  f"V: {melhor_o.vitorias:2d} | D: {melhor_o.derrotas:2d} | E: {melhor_o.empates:2d}")
            print(f"  Media X   -> Fitness: {sum(r.fitness for r in populacao_x)/len(populacao_x):6.1f}")
            print(f"  Media O   -> Fitness: {sum(r.fitness for r in populacao_o)/len(populacao_o):6.1f}")
        
        # Salva checkpoints
        if geracao % salvar_cada == 0:
            os.makedirs('modelos', exist_ok=True)
            melhor_x.salvar(f'modelos/melhor_x_gen{geracao}.json')
            melhor_o.salvar(f'modelos/melhor_o_gen{geracao}.json')
            if verbose:
                print(f"  [SALVO] Modelos salvos em modelos/melhor_x_gen{geracao}.json")
        
        # Evolui para proxima geracao
        if geracao < GERACOES:
            populacao_x = proxima_geracao(populacao_x)
            populacao_o = proxima_geracao(populacao_o)
    
    print(f"\n{'=' * 70}")
    print("  TREINAMENTO CONCLUIDO!")
    print(f"{'=' * 70}")
    
    # Salva modelos finais
    os.makedirs('modelos', exist_ok=True)
    melhor_x.salvar('modelos/melhor_x_final.json')
    melhor_o.salvar('modelos/melhor_o_final.json')
    print("\n  Modelos finais salvos em:")
    print("    -> modelos/melhor_x_final.json")
    print("    -> modelos/melhor_o_final.json")
    
    return melhor_x, melhor_o, populacao_x, populacao_o


# =============================================================================
# JOGAR CONTRA A REDE NEURAL (MODO HUMANO)
# =============================================================================

def jogar_contra_ia(rede_ia, jogador_humano=1):
    """
    Permite um humano jogar contra a rede neural treinada
    """
    jogo = JogoVelha()
    
    print("\n" + "=" * 50)
    print("  VOCE vs REDE NEURAL")
    print("=" * 50)
    print("\nPosicoes do tabuleiro:")
    print("  0 | 1 | 2")
    print("  3 | 4 | 5")
    print("  6 | 7 | 8")
    print()
    
    jogador_atual = 1  # X sempre comeca
    
    while True:
        print(jogo)
        
        if jogador_atual == jogador_humano:
            # Vez do humano
            while True:
                try:
                    pos = int(input("Sua jogada (0-8): "))
                    if pos < 0 or pos > 8:
                        print("Posicao invalida! Use 0-8.")
                        continue
                    if jogo.tabuleiro[pos] != 0:
                        print("Posicao ocupada! Escolha outra.")
                        continue
                    break
                except ValueError:
                    print("Digite um numero entre 0 e 8.")
            
            jogo.jogar(pos, jogador_humano)
        else:
            # Vez da IA
            print("IA esta pensando...")
            movimento = rede_ia.escolher_movimento(jogo.tabuleiro, -jogador_humano)
            jogo.jogar(movimento, -jogador_humano)
            print(f"IA jogou na posicao {movimento}")
        
        resultado = jogo.verificar_vencedor()
        if resultado is not None:
            print(jogo)
            if resultado == jogador_humano:
                print("\n[PARABENS] Voce venceu!")
            elif resultado == 0:
                print("\n[EMPATE]")
            else:
                print("\n[A IA VENCEU] Melhor sorte na proxima.")
            break
        
        jogador_atual *= -1


def simular_partida_visual(rede_x, rede_o, delay=0.5):
    """
    Simula uma partida entre duas redes com visualizacao no terminal
    """
    import time
    jogo = JogoVelha()
    jogador_atual = 1
    
    print("\n" + "=" * 40)
    print("  SIMULACAO: REDE X vs REDE O")
    print("=" * 40)
    
    while True:
        print(jogo)
        time.sleep(delay)
        
        if jogador_atual == 1:
            movimento = rede_x.escolher_movimento(jogo.tabuleiro, 1)
            print(f"  -> Rede X joga em {movimento}")
        else:
            movimento = rede_o.escolher_movimento(jogo.tabuleiro, -1)
            print(f"  -> Rede O joga em {movimento}")
        
        jogo.jogar(movimento, jogador_atual)
        
        resultado = jogo.verificar_vencedor()
        if resultado is not None:
            print(jogo)
            if resultado == 1:
                print("\n[VENCEDOR] REDE X!")
            elif resultado == -1:
                print("\n[VENCEDOR] REDE O!")
            else:
                print("\n[EMPATE]")
            break
        
        jogador_atual *= -1
        print()


# =============================================================================
# MENU PRINCIPAL
# =============================================================================

def menu():
    print("\n" + "=" * 60)
    print("  JOGO DA VELHA COM REDES NEURAIS")
    print("=" * 60)
    print()
    print("  1. Treinar novas redes neurais (modo evolutivo)")
    print("  2. Jogar contra uma rede neural treinada")
    print("  3. Simular partida entre duas redes treinadas")
    print("  4. Sair")
    print()
    
    escolha = input("Escolha uma opcao (1-4): ").strip()
    
    if escolha == '1':
        print("\nIniciando treinamento evolutivo...")
        print("Isso pode levar alguns minutos dependendo das configuracoes.\n")
        melhor_x, melhor_o, _, _ = treinar()
        
        print("\nDeseja jogar contra a melhor rede treinada?")
        if input("(s/n): ").lower() == 's':
            jogar_contra_ia(melhor_x if random.random() > 0.5 else melhor_o)
    
    elif escolha == '2':
        if not os.path.exists('modelos/melhor_x_final.json'):
            print("\n[!] Nenhum modelo treinado encontrado!")
            print("   Treine primeiro (opcao 1) ou coloque modelos na pasta 'modelos/'.")
            return
        
        print("\nCarregando modelos...")
        rede_x = RedeNeural.carregar('modelos/melhor_x_final.json')
        rede_o = RedeNeural.carregar('modelos/melhor_o_final.json')
        
        print("\nQual rede voce quer enfrentar?")
        print("  1. Melhor Rede X")
        print("  2. Melhor Rede O")
        rede_escolha = input("Escolha (1-2): ").strip()
        
        if rede_escolha == '1':
            jogar_contra_ia(rede_x, jogador_humano=-1)
        else:
            jogar_contra_ia(rede_o, jogador_humano=1)
    
    elif escolha == '3':
        if not os.path.exists('modelos/melhor_x_final.json'):
            print("\n[!] Nenhum modelo treinado encontrado!")
            print("   Treine primeiro (opcao 1).")
            return
        
        rede_x = RedeNeural.carregar('modelos/melhor_x_final.json')
        rede_o = RedeNeural.carregar('modelos/melhor_o_final.json')
        simular_partida_visual(rede_x, rede_o)
    
    elif escolha == '4':
        print("\nAte logo!")
        return
    
    else:
        print("\nOpcao invalida!")
    
    print()
    menu()


# =============================================================================
# EXECUCAO
# =============================================================================

if __name__ == '__main__':
    menu()

