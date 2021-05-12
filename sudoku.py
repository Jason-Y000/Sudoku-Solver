import random

class Sudoku:
    def __init__(self,mode,seed):
        self.mode = mode #Different modes will provide different number of clues
        self.seed = seed
        self.ans = []
        self.game = []

    def generate_sol(self):
        board = [] # Make the board a list of lists
        random.seed(self.seed)

############ Generate the solution ################

        for r in range(0,9):
            rows = []
            nums_r = [1,2,3,4,5,6,7,8,9]
            c = 0
            while (c < 9):
                chosen = False
                nums_c = nums_r[:]
                while (chosen == False):
                    if (r == 0):
                        chosen = True
                    if (nums_c == []):
                        number = -1
                        chosen = True
                    else:
                        number = random.choice(nums_c)
                        chosen = True
                        # Check no repeat in column
                        for p in range(0,r):
                            if (board[p][c] == number):
                                nums_c.remove(number)
                                chosen = False
                                break
                        if (chosen == False):
                            continue
                        else:
                        # Check no repeat in 3x3 grid
                            if (r < 3):
                                for row in range(0,r):
                                    if (c >= 0) and (c < 3):
                                        for col in range(0,3):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break

                                    elif (c >= 3) and (c < 6):
                                        for col in range(3,6):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break

                                    elif (c >= 6):
                                        for col in range(6,9):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break

                            elif (r >= 3) and (r < 6):
                                for row in range(3,r):
                                    if (c >= 0) and (c < 3):
                                        for col in range(0,3):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break
                                    elif (c >= 3) and (c < 6):
                                        for col in range(3,6):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break
                                    elif (c >= 6):
                                        for col in range(6,9):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break
                            elif (r >= 6):
                                for row in range(6,r):
                                    if (c >= 0) and (c < 3):
                                        for col in range(0,3):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break
                                    elif (c >= 3) and (c < 6):
                                        for col in range(3,6):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break
                                    elif (c >= 6):
                                        for col in range(6,9):
                                            if (board[row][col] == number):
                                                nums_c.remove(number)
                                                chosen = False
                                                break
                                        if (chosen == False):
                                            break

                if (number == -1):
                    rows = []
                    nums_r = [1,2,3,4,5,6,7,8,9]
                    c = 0
                else:
                    rows.append(number)
                    nums_r.remove(number)
                    c += 1

            board.append(rows)
            # print(board)

        self.ans = [x[:] for x in board]

        return True

    def generate_game(self):
############ Generate the game ################
# TO DO: Use backtracking for creating a unique game

        board = [x[:] for x in self.ans]
        positions = []
        for r in range(0,9):
            for c in range(0,9):
                positions.append((r,c))

        if self.mode == "E": # 41 clues
            zeroes = 0 # Set empty cells to 0
            temp = [element for element in positions]
            while (zeroes < 40):
                pos = random.choice(temp)
                val = board[pos[0]][pos[1]]
                board[pos[0]][pos[1]] = 0

                if (self.countSol(board,0) == 1):
                    positions.remove(pos)
                    temp = [element for element in positions]
                    zeroes += 1
                else:
                    board[pos[0]][pos[1]] = val
                    temp.remove(pos)

        elif self.mode == "M": # 34 clues
            zeroes = 0
            temp = [element for element in positions]
            while (zeroes < 47):
                pos = random.choice(temp)
                val = board[pos[0]][pos[1]]
                board[pos[0]][pos[1]] = 0

                if (self.countSol(board,0) == 1):
                    positions.remove(pos)
                    temp = [element for element in positions]
                    zeroes += 1
                else:
                    board[pos[0]][pos[1]] = val
                    temp.remove(pos)

        elif self.mode == "H": # 24 clues
            zeroes = 0
            temp = [element for element in positions]
            while (zeroes < 57):
                pos = random.choice(temp)
                val = board[pos[0]][pos[1]]
                board[pos[0]][pos[1]] = 0

                if (self.countSol(board,0) == 1):
                    positions.remove(pos)
                    temp = [element for element in positions]
                    zeroes += 1
                else:
                    board[pos[0]][pos[1]] = val
                    temp.remove(pos)

                    if len(temp) == 0:
                        zeroes = 0
                        board = [x[:] for x in self.ans]
                        positions = []
                        for r in range(0,9):
                            for c in range(0,9):
                                positions.append((r,c))
                        temp = [element for element in positions]

        elif self.mode == "EH": # 17 clues
            zeroes = 0
            temp = [element for element in positions]
            while (zeroes < 64):
                pos = random.choice(temp)
                val = board[pos[0]][pos[1]]
                board[pos[0]][pos[1]] = 0

                if (self.countSol(board,0) == 1):
                    positions.remove(pos)
                    temp = [element for element in positions]
                    zeroes += 1
                else:
                    board[pos[0]][pos[1]] = val
                    temp.remove(pos)

                    if len(temp) == 0:
                        zeroes = 0
                        board = [x[:] for x in self.ans]
                        positions = []
                        for r in range(0,9):
                            for c in range(0,9):
                                positions.append((r,c))
                        temp = [element for element in positions]


        self.game = [x[:] for x in board]

        return True

    def solve(self,board):
        # Find empty board location and return True if there are none
        found = False
        for r in range(0,9):
            for c in range(0,9):
                if (board[r][c] == 0):
                    row = r
                    col = c
                    found = True
                if (found == True):
                    break
            if (found == True):
                break
        if (found == False):
            return True
        #####################################

        for n in range(1,10):
            valid_n = True

        ###### Row Check ######
            for c in range(0,9):
                if (board[row][c] == n):
                    valid_n = False
                    break
            if (valid_n == False):
                continue

        ###### Column Check ######
            for r in range(0,9):
                if (board[r][col] == n):
                    valid_n = False
                    break
            if (valid_n == False):
                continue

        ###### 3x3 Grid check ######
            if (row < 3):
                for r in range(0,3):
                    if (col >= 0) and (col < 3):
                        for c in range(0,3):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 3) and (col < 6):
                        for c in range(3,6):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 6):
                        for c in range(6,9):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    if (valid_n == False):
                        break

            elif (row >= 3) and (row < 6):
                for r in range(3,6):
                    if (col >= 0) and (col < 3):
                        for c in range(0,3):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 3) and (col < 6):
                        for c in range(3,6):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 6):
                        for c in range(6,9):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    if (valid_n == False):
                        break

            elif (row >= 6):
                for r in range(6,9):
                    if (col >= 0) and (col < 3):
                        for c in range(0,3):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 3) and (col < 6):
                        for c in range(3,6):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 6):
                        for c in range(6,9):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    if (valid_n == False):
                        break
        ###########################################
            if (valid_n == False):
                continue

        ###### Fill in with the valid number ######
            if (valid_n == True):
                board[row][col] = n

                if (self.solve(board) == True):
                    return True

                board[row][col] = 0

        return False

    def countSol(self,board,count):
        # Find empty board location and return True if there are none
        found = False
        for r in range(0,9):
            for c in range(0,9):
                if (board[r][c] == 0):
                    row = r
                    col = c
                    found = True
                if (found == True):
                    break
            if (found == True):
                break
        if (found == False):
            return count+1
        #####################################

        for n in range(1,10):
            valid_n = True

        ###### Row Check ######
            for c in range(0,9):
                if (board[row][c] == n):
                    valid_n = False
                    break
            if (valid_n == False):
                continue

        ###### Column Check ######
            for r in range(0,9):
                if (board[r][col] == n):
                    valid_n = False
                    break
            if (valid_n == False):
                continue

        ###### 3x3 Grid check ######
            if (row < 3):
                for r in range(0,3):
                    if (col >= 0) and (col < 3):
                        for c in range(0,3):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 3) and (col < 6):
                        for c in range(3,6):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 6):
                        for c in range(6,9):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    if (valid_n == False):
                        break

            elif (row >= 3) and (row < 6):
                for r in range(3,6):
                    if (col >= 0) and (col < 3):
                        for c in range(0,3):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 3) and (col < 6):
                        for c in range(3,6):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 6):
                        for c in range(6,9):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    if (valid_n == False):
                        break

            elif (row >= 6):
                for r in range(6,9):
                    if (col >= 0) and (col < 3):
                        for c in range(0,3):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 3) and (col < 6):
                        for c in range(3,6):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    elif (col >= 6):
                        for c in range(6,9):
                            if (board[r][c] == n):
                                valid_n = False
                                break

                    if (valid_n == False):
                        break
        ###########################################
            if (valid_n == False):
                continue

        ###### Fill in with the valid number ######
            if (valid_n == True):
                board[row][col] = n

                count = self.countSol(board,count)

                board[row][col] = 0

        return count

    def display(self,board):
        for r in range(0,9):
            for c in range(0,9):
                if (c == 2) or (c == 5):
                    print("{} | ".format(board[r][c]),end="")
                elif (c == 8):
                    print(board[r][c])
                else:
                    print("{} ".format(board[r][c]),end="")
            if (r == 2) or (r == 5):
                print("---------------------")
        return True
