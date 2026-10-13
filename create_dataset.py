#%%
import chess
import chess.engine
import random
import json

engine = chess.engine.SimpleEngine.popen_uci('stockfish-windows-x86-64-avx2.exe')

database = {}

for i in range(10):

    board = chess.Board()
    while not board.is_game_over():
        result = engine.analyse(board, chess.engine.Limit(depth=16))
        score = result['score'].pov(chess.WHITE).score(mate_score = 10000)
        move = random.choice(list(board.legal_moves))
        board.push(move)
        print(score)
        if board.fen() not in database.keys():
            database[board.fen()] = score

engine.quit()

#%%

with open(r'scores_dataset/ds1.json', 'w') as f:
    json.dump(database, f)