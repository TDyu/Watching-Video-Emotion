from tkinter import *
import tkinter.font


class MainScreen(Frame):

    def __init__(self, master):
        Frame.__init__(self, master)
        self.grid()
        self.quitButton()
        self.emotion_photos()
        self.text()
        
    def text(self):

        ft = tkinter.font.Font(family='Fixdsys', size=40,
                               weight=tkinter.font.BOLD)

        Label(text='請選擇觀片後的情緒，可複選', font=ft).grid(
            row=0, column=1, columnspan=6, sticky=W+E)
        Label(text='(Please select the emotions after viewing the film)',
              font=ft).grid(row=2, column=1, columnspan=6, sticky=W+E)
        Label(text='步驟1 請先按字母', font=ft).grid(
            row=4, column=1, columnspan=6, sticky=W+E)
        Label(text='(Step 1: Please press the letters)', font=ft).grid(
            row=6, column=1, columnspan=6, sticky=W+E)
        

    def emotion_photos(self):

        ft = tkinter.font.Font(family='Fixdsys', size=25, weight=tkinter.font.BOLD)

        photo_happy = PhotoImage(file="./emotion/happy.png")
        self.label = Label(image=photo_happy)
        self.label.image = photo_happy
        self.label.grid(row=200, column=1)
        Label(text='A', font=ft).grid(row=201, column=1, sticky=W+E+N+S, padx=1, pady=3)

        photo_surprise = PhotoImage(file="./emotion/surprise.png")
        self.label = Label(image=photo_surprise)
        self.label.image = photo_surprise
        self.label.grid(row=200, column=2)
        Label(text='S', font=ft).grid(row=201, column=2, sticky=W+E+N+S)

        photo_afraid = PhotoImage(file="./emotion/afraid.png")
        self.label = Label(image=photo_afraid)
        self.label.image = photo_afraid
        self.label.grid(row=200, column=3)
        Label(text='D', font=ft).grid(row=201, column=3, sticky=W+E+N+S)

        photo_sad = PhotoImage(
            file="./emotion/sad.png")
        self.label = Label(image=photo_sad)
        self.label.image = photo_sad
        self.label.grid(row=200, column=4)
        Label(text='F', font=ft).grid(row=201, column=4, sticky=W+E+N+S)

        photo_angry = PhotoImage(
            file="./emotion/angry.png")
        self.label = Label(image=photo_angry)
        self.label.image = photo_angry
        self.label.grid(row=200, column=5)
        Label(text='G', font=ft).grid(row=201, column=5, sticky=W+E+N+S)

        photo_disgust = PhotoImage(
            file="./emotion/disgust.png")
        self.label = Label(image=photo_disgust)
        self.label.image = photo_disgust
        self.label.grid(row=200, column=6)
        Label(text='H', font=ft).grid(row=201, column=6, sticky=W+E+N+S)

       
    #def emotion_degree(self):
        ft = tkinter.font.Font(family='Fixdsys', size=40, weight=tkinter.font.BOLD)
        Label(text='步驟2 請選擇情緒程度0~5(低~高)，請選擇數字', font=ft).grid(row=203, column=1, columnspan=20, sticky=W+E)
        Label(text='Step 2 (Please choose the number 0 to 5)', font=ft).grid(row=206, column=1, columnspan=20, sticky=W+E)
    
    def quitButton(self):
        ## Provide a quit button to exit the program
        self.quitFrame = Frame(self.master, width=50, height=50)
        self.quitFrame.grid(row=208, column=4, sticky=W+E+N+S)
        self.quitButton = Button(self.quitFrame, text="Quit", command=exit)
        self.quitButton.grid()

    


if __name__ == '__main__':

    
    list = []

    def get_text(event):
        list.append(event.char)
        text.delete('0.0', END)
        text.insert(END, ''.join(list))
        #canv.create_text(50, 50, text=text.get("0.0", END), anchor=W, width=100)
    
    root = Tk()
    main = MainScreen(root)
    root.overrideredirect(True)
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(),
                                       root.winfo_screenheight()))


    def A_Key(event):
        print("A pressed")
        list.append('A')

    def S_Key(event):
        print("S pressed")
        list.append('S')

    def D_Key(event):
        print("D pressed")
        list.append('D')

    def F_Key(event):
        print("F pressed")
        list.append('F')

    def G_Key(event):
        print("G pressed")
        list.append('G')

    def H_Key(event):
        print("H pressed")
        list.append('H')

    def One_Key(event):
        print("1")
        list.append('1')

    def Two_Key(event):
        print("2")
        list.append('2')

    def Three_Key(event):
        print("3")
        list.append('3')

    def Four_Key(event):
        print("4")
        list.append('4')

    def Five_Key(event):
        print("5")
        list.append('5')

    root.focus_set()
    root.bind('A', A_Key)
    root.bind('S', S_Key)
    root.bind('D', D_Key)
    root.bind('F', F_Key)
    root.bind('G', G_Key)
    root.bind('H', H_Key)

    root.bind('1', One_Key)
    root.bind('2', Two_Key)
    root.bind('3', Three_Key)
    root.bind('4', Four_Key)
    root.bind('5', Five_Key)
    root.grid()

    canv = Canvas(root, width=50, height=100)
    canv.grid(row=210, columnspan=3, sticky=W+E+N+S)
    text = Text(canv)
    canv.focus_set()
    canv.bind("<Key>", get_text)
      
    root.mainloop()
    
