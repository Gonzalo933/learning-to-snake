import numpy as np


def preprocess_board(board, downsampling):
    ###### DEBUG SPACE
    # import matplotlib.pyplot as plt
    #
    # plt.title("ASD")
    # plt.imshow(board)
    # plt.show()
    # breakpoint()
    #####
    board = board[::downsampling, ::downsampling, :]  # downsample by factor of 2
    grayscale_board = rgb2gray(board)
    # processed_board = np.copy(board)
    # processed_board[board[:, :, 2] == 255] = food_color  # Food
    # processed_board[board[:, :, 0] == 255] = snake_head_color  # Snake head
    # processed_board[board[:, :, 0] == 1] = snake_torso_color  # Snake torso
    # processed_board[board[:, :, 1] == 255] = background_color  # Background
    # processed_board = processed_board[:, :, 0]
    # Scale between 0 and 1
    processed_board = (grayscale_board - grayscale_board.min()) / (
        grayscale_board.max() - grayscale_board.min()
    )
    return processed_board.astype(np.float).ravel()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
