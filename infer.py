import random
import cv2
import numpy as np
import time
import ctypes


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function {func.__name__} took {elapsed_time:.8f} seconds to execute.")
        return result
    return wrapper


class Infer:
    net = cv2.dnn.readNetFromONNX("mouse.onnx")
    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_node = []

    @timeit
    def infer(self, node1:tuple, node2:tuple):
        """
        :param node1:   当前坐标x,y
        :param node2:   目标坐标x,y
        :return:
        """
        dx = node2[0] - node1[0]
        dy = node2[1] - node1[1]
        matblob = np.array([[dx,dy]])
        self.net.setInput(matblob)
        output = self.net.forward()
        output_list = output.tolist()[0]
        print(f'output_list = {output_list}')
        return output_list

    def show_result(self, canvas: tuple, start_node: tuple, end_node: tuple):
        def mouse_callback(event, x, y, flags, userdata):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_x = x
                self.mouse_y = y

        node_result1 = self.infer(start_node, end_node)

        for i in node_result1:
            _x = int(i[0]) + start_node[0]
            _y = int(i[1]) + start_node[1]
            self.mouse_node.append((_x, _y))
        self.mouse_node.append(end_node)
        window_title = 'show_result'
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_title, mouse_callback)
        _img = np.zeros((canvas[0], canvas[1], 3), np.uint8) + 255
        while True:
            img = _img.copy()
            for i in self.mouse_node:
                _x = i[0]
                _y = i[1]
                cv2.circle(img, (_x, _y),5, (111, 0, 0), -1)

            cv2.circle(img, (start_node[0], start_node[1]),5, (0, 0, 255), -1)
            cv2.circle(img, (end_node[0], end_node[1]),5, (255, 0, 0), -1)
            cv2.putText(img, f'{self.mouse_x}, {self.mouse_y}', (self.mouse_x, self.mouse_y), cv2.FONT_ITALIC, 1.5, (0, 255, 0), 3)
            cv2.imshow(window_title, img)
            key = cv2.waitKey(1)
            if key == 27:
                break   # 按下ESC 关闭窗口
        cv2.destroyWindow(window_title)
        time.sleep(1)
        self.move()

    def move(self):
        print(self.mouse_node)
        # 调用Windows API移动鼠标
        for node in self.mouse_node:
            ctypes.windll.user32.SetCursorPos(node[1], node[0])
            delay = round(random.uniform(0.001, 0.005), 6)
            time.sleep(delay)


if __name__ == '__main__':
    canvas_size = 600, 1200
    start_node = 100, 30
    end_node = 600, 200     #

    Infer().show_result(canvas_size, start_node, end_node)