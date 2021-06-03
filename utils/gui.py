from tkinter import Tk, TOP, BOTH, X, Y, N, W, E, S, N, LEFT, RIGHT, END, YES, NO, SUNKEN, ALL, VERTICAL, HORIZONTAL, BOTTOM, CENTER
from tkinter import FIRST, LAST, ROUND, NONE, DISABLED
from tkinter import Text, Canvas, Listbox, Scale, Checkbutton, Label, Entry, Scrollbar, Frame, Button, Spinbox
from tkinter import BooleanVar, StringVar# , IntVar, DoubleVar
from tkinter import Toplevel, TclError, filedialog, messagebox
from utils.param_ops import less_kwargs, more_kwargs
# from tkinter.ttk import Frame, Label, Entry, Button #- old fashion
from collections import namedtuple
from itertools import count

Checkboxes = namedtuple('Checkboxes', 'ckb, var')
Entries = namedtuple('Entries', 'lbl, etr, var, color')
CheckEntries = namedtuple('CheckEntries', 'ckb, etr, bvar, svar, color')

def __checkbox(panel, text, value, callback, gui_kwargs):
    var = BooleanVar(panel)
    gui = Checkbutton(panel, text = text, variable = var, command = callback)
    var.set(value)
    gui.var = var # delete?
    return gui, var

def make_checkbox(panel, text, value, callback, gui_kwargs, control = 1):
    gui, var = __checkbox(panel, text, value, callback, gui_kwargs)
    if control > 0: # 0 line for raw combination
        gui.pack(side = TOP, anchor = W, **gui_kwargs) # sticky = W is special for gird
    return Checkboxes(gui, var)

def __entry(panel, value, callback, gui_kwargs):
    char_width = less_kwargs(gui_kwargs, 'char_width', None)
    prompt_str = less_kwargs(gui_kwargs, 'prompt_str', str(value) + "  ")
    var = StringVar(panel)
    common_args = dict(textvariable = var, width = char_width, justify = CENTER)
    if isinstance(value, tuple):
        value, start, end, inc = value
        gui = Spinbox(panel, from_ = start, to = end, increment = inc, **common_args)
        def spin_click(*event):
            callback(None) # ???
        # gui.bind('<Button-1>', lambda *e: callback(None))
        var.trace('w', spin_click)
    else:
        gui = Entry(panel, **common_args)
    var.set(value) # even no initial value?
    default_color = gui.cget('highlightbackground')
    gui.bind('<KeyRelease>', callback)
    # gui.var = var # delete ?
    def on_entry_click(event):
        if gui.get() == prompt_str:
            gui.delete(0, "end") # delete all the text in the entry
            gui.insert(0, '')    # insert blank for user input
            gui.config(fg = 'black')
    def on_focusout(event):
        if gui.get() == '':
            gui.insert(0, prompt_str)
            gui.config(fg = 'grey')
        else:
            callback(None)
    gui.bind('<FocusIn>', on_entry_click)
    gui.bind('<FocusOut>', on_focusout)
    if value:
        on_entry_click(None)
    else:
        on_focusout(None)
    return gui, var, default_color

def make_entry(panel, text, value, callback, gui_kwargs, control = 1):
    pnl = Frame(panel)
    if isinstance(control, dict):
        gui_kwargs.update(control)
        control = 1
    elif less_kwargs(gui_kwargs, 'char_width', None) is None:
        gui_kwargs['char_width'] = 4 if control == 1 else 20
    gui, var, clr = __entry(pnl, value, callback, gui_kwargs)
    lbl = Label(pnl, text = text)
    pnl.pack(side = TOP, fill = BOTH, **gui_kwargs)
    if control == 1:
        lbl.pack(side = LEFT,  anchor = W, fill = X, expand = YES)
        gui.pack(side = RIGHT, anchor = E)
    else: # 2 lines
        lbl.pack(side = TOP, anchor = W)
        gui.pack(side = TOP, anchor = E, expand = YES, fill = X)
    return Entries(lbl, gui, var, clr)

def make_checkbox_entry(panel, text, values, callbacks, gui_kwargs, control = 2):
    # e.g. (panel, 'curve', (True, 'x'), (func1, func2), {char_width:3, padx:...})
    ckb_value,    etr_value    = values
    ckb_callback, etr_callback = callbacks
    pnl = Frame(panel)
    pnl.pack(side = TOP, fill = X, anchor = W, **gui_kwargs)
    if less_kwargs(gui_kwargs, 'char_width', None) is None:
        gui_kwargs['char_width'] = 4 if control == 1 else 20
    etr, svar, clr = __entry(pnl,                  etr_value, etr_callback, gui_kwargs)
    ckb, bvar   = __checkbox(pnl, 'Apply ' + text, ckb_value, ckb_callback, gui_kwargs)
    if control == 1:
        wht = Label(pnl)
        ckb.pack(side = LEFT, anchor = W)
        wht.pack(side = LEFT,  fill = X, expand = YES)
        etr.pack(side = RIGHT, fill = X)
    else: # 2 lines
        ckb.pack(side = TOP, anchor = W)
        etr.pack(side = TOP, fill = X, anchor = E, expand = YES)
    return CheckEntries(ckb, etr, bvar, svar, clr)

def get_checkbox(ckbxes, ctype = 0):
    if ctype == 0:
        gen = (v.get() for _, v in ckbxes)
    else:
        gen = (v.get() for _, _, v, _, _ in ckbxes)
    return ckbxes.__class__(*gen)

def get_entry(entries, entry_dtypes, fallback_values, ctype = 0):
    gen = zip(entries, entry_dtypes, fallback_values)
    res = []
    if ctype == 0:
        for (l, g, v, c), d, f in gen:
            try:
                res.append(d(v.get()))
                g.config(highlightbackground = c)
            except Exception as e:
                print(l.cget('text'), e, 'use', f, 'instead')
                g.config(highlightbackground = 'pink')
                res.append(f)
    else:
        for (b, g, _, v, c), d, f in gen:
            try:
                t = d(v.get())
                if d is eval:
                    t(0.5)
                res.append(t)
                g.config(highlightbackground = c)
            except Exception as e:
                print(b.cget('text'), e, 'use', f, 'instead')
                g.config(highlightbackground = 'pink')
                res.append(f)
    if entries.__class__ is tuple:
        return tuple(res)
    return entries.__class__(*res)

def make_namedtuple_gui(make_func, panel, values, callback, control = None, **gui_kwargs):
    if control is None:
        return values.__class__(
            *(make_func(panel, n.replace('_', ' ').title(), v, callback, gui_kwargs.copy()) for n, v in zip(values._fields, values))
            )
    widgets = []
    for n, v, c in zip(values._fields, values, control):
        w = make_func(panel, n.replace('_', ' ').title(), v, callback, gui_kwargs.copy(), c)
        widgets.append(w)
    return values.__class__(*widgets)

    # demo_func = less_kwargs(gui_kwargs, 'demo_func', 'lambda x:x')
    # entry.pack(side = TOP, anchor = W, **gui_kwargs)
    # def on_entry_click(event):
    #     if entry.get() == demo_func:
    #         entry.delete(0, "end") # delete all the text in the entry
    #         entry.insert(0, 'x')    # Insert blank for user input
    #         entry.config(fg = 'black')

    # def on_focusout(event):
    #     if entry.get().strip() in ('', 'x'):
    #         entry.delete(0, "end")
    #         entry.insert(0, demo_func)
    #         entry.config(fg = 'grey')
    # entry.bind('<FocusIn>', on_entry_click)
    # entry.bind('<FocusOut>', on_focusout)
    # entry.config(fg = 'grey')

import numpy as np
def bezier_curve(canvas, center, length, top, bottom, func = lambda x: x, num_points = 5, **draw_kwargs):
    assert num_points > 2 and num_points % 2 # even
    t_point = np.asarray([center, top])
    b_point = np.asarray([center, bottom])
    c_point = np.asarray([center + length, 0.5 * (top + bottom)])
    
    coord = []
    for i in range(num_points):
        ratio = i / (num_points - 1)
        l_ratio = func(ratio)
        r_ratio = func(1 - ratio)
        l = t_point * l_ratio + c_point * (1 - l_ratio)
        r = b_point * r_ratio + c_point * (1 - r_ratio)
        point = l * ratio + r * (1 - ratio)
        coord.extend(point)
        if i << 1 == num_points - 1: # 5 == 2 * 2 + 1
            mid_point = point
    canvas.create_line(*coord, smooth = True, **draw_kwargs)# outline = '#f11', fill = '#1f1', width = 2)#, start=30)#, extent=120, style=tk.ARC, width=3)
    return mid_point

import platform
OS = platform.system()

class _MousewheelSupport(object):

    # implemetation of singleton pattern
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, root, horizontal_factor=2, vertical_factor=2):

        self._active_area = None

        if isinstance(horizontal_factor, int):
            self.horizontal_factor = horizontal_factor
        else:
            raise Exception("Vertical factor must be an integer.")

        if isinstance(vertical_factor, int):
            self.vertical_factor = vertical_factor
        else:
            raise Exception("Horizontal factor must be an integer.")

        if OS == "Linux":
            root.bind_all('<4>', self._on_mousewheel, add='+')
            root.bind_all('<5>', self._on_mousewheel, add='+')
        else:
            # Windows and MacOS
            root.bind_all("<MouseWheel>", self._on_mousewheel, add='+')

    def _on_mousewheel(self, event):
        if self._active_area:
            self._active_area.onMouseWheel(event)

    def _mousewheel_bind(self, widget):
        self._active_area = widget

    def _mousewheel_unbind(self):
        self._active_area = None

    def add_support_to(self, widget=None, xscrollbar=None, yscrollbar=None, what="units", horizontal_factor=None, vertical_factor=None):
        if xscrollbar is None and yscrollbar is None:
            return

        if xscrollbar is not None:
            horizontal_factor = horizontal_factor or self.horizontal_factor

            xscrollbar.onMouseWheel = self._make_mouse_wheel_handler(widget, 'x', self.horizontal_factor, what)
            xscrollbar.bind('<Enter>', lambda event, scrollbar=xscrollbar: self._mousewheel_bind(scrollbar))
            xscrollbar.bind('<Leave>', lambda event: self._mousewheel_unbind())

        if yscrollbar is not None:
            vertical_factor = vertical_factor or self.vertical_factor

            yscrollbar.onMouseWheel = self._make_mouse_wheel_handler(widget, 'y', self.vertical_factor, what)
            yscrollbar.bind('<Enter>', lambda event, scrollbar=yscrollbar: self._mousewheel_bind(scrollbar))
            yscrollbar.bind('<Leave>', lambda event: self._mousewheel_unbind())

        main_scrollbar = yscrollbar if yscrollbar is not None else xscrollbar

        if widget is not None:
            if isinstance(widget, list) or isinstance(widget, tuple):
                list_of_widgets = widget
                for widget in list_of_widgets:
                    widget.bind('<Enter>', lambda event: self._mousewheel_bind(widget))
                    widget.bind('<Leave>', lambda event: self._mousewheel_unbind())

                    widget.onMouseWheel = main_scrollbar.onMouseWheel
            else:
                widget.bind('<Enter>', lambda event: self._mousewheel_bind(widget))
                widget.bind('<Leave>', lambda event: self._mousewheel_unbind())

                widget.onMouseWheel = main_scrollbar.onMouseWheel

    @staticmethod
    def _make_mouse_wheel_handler(widget, orient, factor=1, what="units"):
        view_command = getattr(widget, orient + 'view')

        if OS == 'Linux':
            def onMouseWheel(event):
                if event.num == 4:
                    view_command("scroll", (-1) * factor, what)
                elif event.num == 5:
                    view_command("scroll", factor, what)

        elif OS == 'Windows':
            def onMouseWheel(event):
                view_command("scroll", (-1) * int((event.delta / 120) * factor), what)

        elif OS == 'Darwin':
            def onMouseWheel(event):
                view_command("scroll", event.delta, what)

        return onMouseWheel


class ScrollingArea(Frame, object):

    def __init__(self,
                 master,
                 width = None,
                 height = None,
                 anchor = N,
                 scroll_vertically = True,
                 background = None, show_scrollbar = False, inner_frame_cls = Frame, **kw):
        Frame.__init__(self, master, class_="Scrolling_Area", background=background)
        # self.grid_columnconfigure(0, weight=1)
        # self.grid_rowconfigure(0, weight=1)

        self._width = width
        self._height = height

        self._canvas = Canvas(self, background = background, highlightthickness = 0, width = width, height = height)
        self._canvas.pack(side = LEFT)
        # self.canvas.grid(row=0, column=0, sticky=N + E + W + S)

        self._scrollbar = Scrollbar(self, orient = VERTICAL if scroll_vertically else HORIZONTAL)
        if show_scrollbar:
            self._scrollbar.pack()#row=0, column=1, sticky=N + S)

        self._canvas.configure(yscrollcommand = self._scrollbar.set)
        self._scrollbar['command'] = self._canvas.yview

        # self.rowconfigure(0, weight=1)
        # self.columnconfigure(0, weight=1)

        self._inner_frame = inner_frame_cls(self._canvas, **kw)
        self._inner_frame.pack(anchor = anchor)

        self._canvas.create_window(0, 0, window = self._inner_frame, anchor='nw', tags="inner_frame")

        self._canvas.bind('<Configure>', self._on_canvas_configure)

        if scroll_vertically:
            _MousewheelSupport(self).add_support_to(self._canvas, yscrollbar = self._scrollbar)
        else:
            _MousewheelSupport(self).add_support_to(self._canvas, xscrollbar = self._scrollbar)

    @property
    def width(self):
        return self._canvas.winfo_width()

    @width.setter
    def width(self, width):
        self._canvas.configure(width=width)

    @property
    def height(self):
        return self._canvas.winfo_height()

    @height.setter
    def height(self, height):
        self._canvas.configure(height=height)

    @property
    def inner_frame(self):
        return self._inner_frame

    def set_size(self, width, height):
        self._canvas.configure(width=width, height=height)

    def _on_canvas_configure(self, event):
        width  = max(self._inner_frame.winfo_reqwidth(), event.width)
        height = max(self._inner_frame.winfo_reqheight(), event.height)

        self._canvas.configure(scrollregion="0 0 %s %s" % (width, height))
        self._canvas.itemconfigure("inner_frame", width=width, height=height)

    def update_viewport(self):
        self.update()

        window_width  = self._inner_frame.winfo_reqwidth()
        window_height = self._inner_frame.winfo_reqheight()

        if self._width is None:
            canvas_width = window_width
        else:
            canvas_width = min(self._width, window_width)

        if self._height is None:
            canvas_height = window_height
        else:
            canvas_height = min(self._height, window_height)

        self._canvas.configure(scrollregion="0 0 %s %s" % (window_width, window_height), width=canvas_width, height=canvas_height)
        self._canvas.itemconfigure("inner_frame", width=window_width, height=window_height)