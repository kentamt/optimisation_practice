list = [4,10,1,5]



def bit_plus(list):

    sum = 0
    for bit in range(1<<len(list)): # 0(0b000)から7(0b111)まで
        print(format(bit, '04b'),format(15-bit, '04b'))
        for i in range(len(list)):
            mask = 1 << i
            if bit&mask: # 右からi番目にビットが立っているかどうか判定
                sum += list[i]
    return sum

 
print(bit_plus(list)) # 出力は60

