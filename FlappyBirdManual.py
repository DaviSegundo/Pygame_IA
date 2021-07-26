import os
import pygame
import random
import neat
from classes.Chao import Chao
from classes.Cano import Cano
from classes.Passaro import Passaro

ai_jogando = True
geracao = 0

TELA_LARGURA = 500
TELA_ALTURA = 800

IMAGEM_BACKGROUND = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bg.png')))

pygame.font.init()
FONTE_PONTOS = pygame.font.SysFont('arial', 35)


def desenhar_tela(tela, passaros, canos, chao, pontos):
    tela.blit(IMAGEM_BACKGROUND, (0, 0))

    chao.desenhar(tela)

    for passaro in passaros:
        passaro.desenhar(tela)

    for cano in canos:
        cano.desenhar(tela)

    texto = FONTE_PONTOS.render(f"Pontuação: {pontos}", 1, (255, 255, 255))
    tela.blit(texto, (TELA_LARGURA - 10 - texto.get_width(), 10))
    
    if ai_jogando:
        texto = FONTE_PONTOS.render(f"Geração: {geracao}", 1, (255, 255, 255))
        tela.blit(texto, (10, 10))

    pygame.display.update()


def main(genomas, config): # fitness function
    # configurações de funcionamento da IA
    global geracao
    geracao += 1

    if ai_jogando:
        redes = []
        lista_genomas = []
        passaros = []

        for _, genoma in genomas:
            rede = neat.nn.FeedForwardNetwork.create(genoma, config)
            redes.append(rede)
            genoma.fitness = 0 # pontuação inicial
            lista_genomas.append(genoma)
            passaros.append(Passaro(230, 350))
    else:
        passaros = [Passaro(230, 350)]

    # configurações iniciais do game

    chao = Chao(730)
    canos = [Cano(700)]
    tela = pygame.display.set_mode((TELA_LARGURA, TELA_ALTURA))
    pygame.display.set_caption('FlappyBird IA')
    pontos = 0
    relogio = pygame.time.Clock()

    # Loop de execução do jogo
    rodando = True
    while rodando:
        relogio.tick(30)

        # interação com o player
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
                pygame.quit()
                quit()
            if not ai_jogando:
                if evento.type == pygame.KEYDOWN:
                    if evento.key == pygame.K_SPACE:
                        for passaro in passaros:
                            passaro.pular()

        # Lógica para mostrar a IA para qual distancia do cano ela deve olhar
        indice_cano = 0
        if len(passaros) > 0:
            if len(canos) > 1 and passaros[0].x > ( canos[0].x + canos[0].CANO_TOPO.get_width()):
                indice_cano = 1
        else:
            rodando = False
            break

        # movimento dos objetos na tela
        for i, passaro in enumerate(passaros):
            passaro.mover()
            
            # aumentar fitness do pássaro
            lista_genomas[i].fitness += 0.1

            output = redes[i].activate((passaro.y, 
                                        abs(passaro.y - canos[indice_cano].altura), 
                                        abs(passaro.y - canos[indice_cano].pos_base)))
            if output[0] > 0.5:
                passaro.pular()

        chao.mover()

        # Lógica dos canos no game
        adicionar_cano = False
        remover_canos = []

        for cano in canos:
            for i, passaro in enumerate(passaros):
                if cano.colidir(passaro):
                    passaros.pop(i)
                    if ai_jogando:
                        # penalização por colidir com o cano
                        lista_genomas[i].fitness -= 3
                        lista_genomas.pop(i)
                        redes.pop(i)

                if not cano.passou and passaro.x > cano.x:
                    cano.passou = True
                    adicionar_cano = True
            cano.mover()
            if cano.x + cano.CANO_TOPO.get_width() < 0:
                remover_canos.append(cano)

        if adicionar_cano:
            pontos += 1
            canos.append(Cano(600))
            if ai_jogando:
                for genoma in lista_genomas:
                    genoma.fitness += 5
        for cano in remover_canos:
            canos.remove(cano)

        # Matar o pássaro se ele passar da tela por cima ou por baixo
        for i, passaro in enumerate(passaros):
            if (passaro.y + passaro.imagem.get_height()) > chao.y or passaro.y < 0:
                passaros.pop(i)
                if ai_jogando:
                    lista_genomas.pop(i)
                    redes.pop(i)

        # desenhar a tela
        desenhar_tela(tela, passaros, canos, chao, pontos)

def rodar(path):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                path)
    populacao = neat.Population(config)
    populacao.add_reporter(neat.StdOutReporter(True))
    populacao.add_reporter(neat.StatisticsReporter())

    if ai_jogando:
        populacao.run(main, 50)
    else:
        main(None, None)

if __name__ == '__main__':
    path = 'config.txt'
    rodar(path)