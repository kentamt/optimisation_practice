import itertools
import numpy as np 
import copy 

def set_queen(board, ir, ic):
    s = board.shape[0]

    board[ir, :] = 0
    board[:, ic] = 0

    i = 0
    while True:
        if not (0 <= ir+i <= s-1 and 0 <= ic+i <= s-1):
            break
        board[ir+i, ic+i] = 0
        i += 1
    i=0            
    while True:
        if not (0 <= ir-i <= s-1 and 0 <= ic+i <= s-1):
            break
        board[ir-i, ic+i] = 0
        i += 1
    i=0            
    while True:
        if not (0 <= ir+i <= s-1 and 0 <= ic-i <= s-1): 
            break
        board[ir+i, ic-i] = 0
        i += 1

    i = 0
    while True:
        if not (0 <= ir-i <= s-1 and 0 <= ic-i <= s-1):
            break
        board[ir-i, ic-i] = 0
        i += 1


if __name__ == "__main__":
    
    
    n_columns = 16
    solutions = [] # クイーンの位置のタプル(ir, ic)
    
    solutions.append((0, 0))
    
    def check_column(ic):

        # 端まで到達していたら抜ける
        if ic == n_columns:
            return ic

        # 盤面をセットアップ
        board = np.ones((n_columns, n_columns), dtype=int)
        for _ir, _ic in solutions:
            set_queen(board, _ir, _ic)

        # 注目している列をとりだす
        c = board[:, ic]

        # 利用可能なセルについて再帰的に処理をすすめる
        for ir in range(n_columns):
            if c[ir] == 1:
                # 利用可能なセルなので解に追加
                solutions.append((ir, ic))
                # ひとつ右の列に移動する
                ic += 1
                ic = check_column(ic)
                
                # もし右端に到達していれば抜ける                
                if ic == n_columns:
                    return ic
        
        # 利用可能なセルがない場合、解をひとつへらす
        solutions.pop(-1)        
        
        # これ以上潜れない場合は再帰をひとつぬける
        return ic-1

    check_column(1)
    board = np.zeros((n_columns, n_columns), dtype=int)
    for ir, ic in solutions:
        board[ir, ic] = 1
    print(solutions)        
    print(board)
    
    