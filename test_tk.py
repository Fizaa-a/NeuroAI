import tkinter as tk
from tkinter import messagebox

def diag():
    root = tk.Tk()
    root.title("Mac Diagnostic Window")
    root.geometry("400x300")
    
    label = tk.Label(root, text="If you can see this, \nTkinter is working correctly.", 
                    font=("Helvetica", 16), pady=50)
    label.pack()
    
    btn = tk.Button(root, text="Click Me", command=lambda: messagebox.showinfo("Success", "Buttons work!"))
    btn.pack()
    
    root.lift()
    root.mainloop()

if __name__ == "__main__":
    diag()
