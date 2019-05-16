# -*- coding: utf-8 -*-
import tkinter as tk
import tkinter.font

from experiment_logging import ANSWER


class QWindow(tk.Tk):

    def __init__(self, seconds, video_name):
        tk.Tk.__init__(self)
        self.attributes('-fullscreen', True)
        self.grid()

        self.emotion_photos()
        self.notice_text()

        self.remaining = 0
        self.countdown(seconds)

        self.key_event_bind()

        self.set_display_text()
        self.video_name = video_name
        self.answers = {
            'A': 0,
            'S': 0,
            'D': 0,
            'F': 0,
            'G': 0,
            'H': 0}

    def countdown(self, remaining=None):
        if remaining is not None:
            self.remaining = remaining

        if self.remaining <= 0:
            print(str(self.answers))
            ANSWER(self.video_name, self.answers)
            self.destroy()
        else:
            self.label.configure(text="%d" % self.remaining)
            self.remaining = self.remaining - 1
            self.after(1000, self.countdown)

    def notice_text(self):
        ft = tkinter.font.Font(family='Fixdsys', size=40,
                          weight=tk.font.BOLD)

        tk.Label(text='請選擇觀片後的情緒，可複選', font=ft).grid(
            row=0, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='(Please select the emotions after viewing the film)',
              font=ft).grid(row=2, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='步驟1 請先按字母選擇情緒', font=ft).grid(
            row=4, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='(Step 1: Please press the letters)', font=ft).grid(
            row=6, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='步驟2 再請選擇情緒程度0~5(低~高)，請按數字', font=ft).grid(
            row=203, column=3, columnspan=6, sticky=tk.W + tk.E)
        tk.Label(text='Step 2 (Please choose the number 0 to 5)', font=ft).grid(
            row=206, column=3, columnspan=6, sticky=tk.W + tk.E)
        

    def emotion_photos(self):
        ft = tkinter.font.Font(family='Fixdsys', size=25, weight=tk.font.BOLD)

        photo_happy = tk.PhotoImage(file="./emotion/happy.png")
        self.label = tk.Label(image=photo_happy)
        self.label.image = photo_happy
        self.label.grid(row=200, column=2)
        tk.Label(text='A', font=ft).grid(row=201, column=2,
                                         sticky=tk.W + tk.E + tk.N + tk.S)

        photo_surprise = tk.PhotoImage(file="./emotion/surprise.png")
        self.label = tk.Label(image=photo_surprise)
        self.label.image = photo_surprise
        self.label.grid(row=200, column=3)
        tk.Label(text='S', font=ft).grid(
            row=201, column=3, sticky=tk.W + tk.E + tk.N + tk.S)

        photo_afraid = tk.PhotoImage(file="./emotion/afraid.png")
        self.label = tk.Label(image=photo_afraid)
        self.label.image = photo_afraid
        self.label.grid(row=200, column=4)
        tk.Label(text='D', font=ft).grid(
            row=201, column=4, sticky=tk.W + tk.E + tk.N + tk.S)

        photo_sad = tk.PhotoImage(
            file="./emotion/sad.png")
        self.label = tk.Label(image=photo_sad)
        self.label.image = photo_sad
        self.label.grid(row=200, column=5)
        tk.Label(text='F', font=ft).grid(
            row=201, column=5, sticky=tk.W + tk.E + tk.N + tk.S)

        photo_angry = tk.PhotoImage(
            file="./emotion/angry.png")
        self.label = tk.Label(image=photo_angry)
        self.label.image = photo_angry
        self.label.grid(row=200, column=6)
        tk.Label(text='G', font=ft).grid(
            row=201, column=6, sticky=tk.W + tk.E + tk.N + tk.S)

        photo_disgust = tk.PhotoImage(
            file="./emotion/disgust.png")
        self.label = tk.Label(image=photo_disgust)
        self.label.image = photo_disgust
        self.label.grid(row=200, column=7)
        tk.Label(text='H', font=ft).grid(
            row=201, column=7, sticky=tk.W + tk.E + tk.N + tk.S)

    def set_display_text(self):
        ft = tkinter.font.Font(family='Fixdsys', size=20,
                               weight=tk.font.BOLD)
        self.display_label_text = tk.StringVar()
        self.display_label_text.set('已選擇:')
        self.display_label = tk.Label(textvariable=self.display_label_text, font=ft, bg='white').grid(
            row=210, column=5, columnspan=1, sticky=tk.W + tk.E + tk.N + tk.S)

    def A_Key(self, event):
        if not self.has_A:
            self.has_choice = True
            text = self.display_label_text.get() + '\nA   '
            self.display_label_text.set(text)
            self.has_A = True
            self.press = 'A'

    def S_Key(self, event):
        if not self.has_S:
            self.has_choice = True
            text = self.display_label_text.get() + '\nS   '
            self.display_label_text.set(text)
            self.has_S = True
            self.press = 'S'

    def D_Key(self, event):
        if not self.has_D:
            self.has_choice = True
            text = self.display_label_text.get() + '\nD   '
            self.display_label_text.set(text)
            self.has_D = True
            self.press = 'D'

    def F_Key(self, event):
        if not self.has_F:
            self.has_choice = True
            text = self.display_label_text.get() + '\nF   '
            self.display_label_text.set(text)
            self.has_F = True
            self.press = 'F'

    def G_Key(self, event):
        if not self.has_G:
            self.has_choice = True
            text = self.display_label_text.get() + '\nG   '
            self.display_label_text.set(text)
            self.has_G = True
            self.press = 'G'

    def H_Key(self, event):
        if not self.has_H:
            self.has_choice = True
            text = self.display_label_text.get() + '\nH   '
            self.display_label_text.set(text)
            self.has_H = True
            self.press = 'H'

    def One_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   1'
            self.display_label_text.set(text)
            self.answers[self.press] = 1
            self.has_choice = False
            self.press = None

    def Two_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   2'
            self.display_label_text.set(text)
            self.answers[self.press] = 2
            self.has_choice = False
            self.press = None

    def Three_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   3'
            self.display_label_text.set(text)
            self.answers[self.press] = 3
            self.has_choice = False
            self.press = None

    def Four_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   4'
            self.display_label_text.set(text)
            self.answers[self.press] = 4
            self.has_choice = False
            self.press = None

    def Five_Key(self, event):
        if self.has_choice and self.press:
            text = self.display_label_text.get() + '   5'
            self.display_label_text.set(text)
            self.answers[self.press] = 5
            self.has_choice = False
            self.press = None

    def key_event_bind(self):
        self.has_A = False
        self.has_S = False
        self.has_D = False
        self.has_F = False
        self.has_G = False
        self.has_H = False
        self.has_choice = False
        self.press = None

        self.bind('a', self.A_Key)
        self.bind('s', self.S_Key)
        self.bind('d', self.D_Key)
        self.bind('f', self.F_Key)
        self.bind('g', self.G_Key)
        self.bind('h', self.H_Key)

        self.bind('1', self.One_Key)
        self.bind('2', self.Two_Key)
        self.bind('3', self.Three_Key)
        self.bind('4', self.Four_Key)
        self.bind('5', self.Five_Key)


if __name__ == '__main__':
    main = QWindow(5, 'test_video')
    main.mainloop()
