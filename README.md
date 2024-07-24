# manim-beamer: Beamer format in manim-slides! :tada:

This small repository is meant to complement [manim-slides](https://github.com/jeertmans/manim-slides), and does not override it. 
Contained within are a few helpful classes to emulate the [LaTeX beamer format](https://www.overleaf.com/learn/latex/Beamer) within manim-slides!

Most notably, this focuses on the Beamer blocks used to highlight important sentences/words (e.g., block, alertblock, examples). 
The manim-beamer classes implementing these Beamer blocks already use the necessary colors to achieve a similar effect.

This repository also provides the user with code in how to implement LaTeX lists with different markers to emulate itemize, enumerate, and so on! 
Of course, these manim-beamer List classes are also compatible with the manim-beamer Block classes as well.

## Demo of an animated slide with blocks and itemized lists
https://github.com/user-attachments/assets/d99e1ce3-c08c-4a0a-a5d8-035d8a577b10

Lastly, code is also provided that allows convenient use of slides containing captioned diagrams or tables. 
For the tables, it is possible to highlight particular rows of interest as well. 

## Demos of animated slideshows with diagrams, tables, graphs, etc. 
https://github.com/user-attachments/assets/63638746-ace4-428e-a92c-8af9dc045c97

https://github.com/user-attachments/assets/107a5977-7136-4dd2-bf06-934e9bc962b6

https://github.com/user-attachments/assets/9dbc917b-4643-4896-8026-2af7a115df69

https://github.com/user-attachments/assets/0424a5ac-eb95-4038-857d-fd67637dfcf4

## Troubleshooting :worried: 
- If you are having trouble with running 'manim-slides' command with the 'mbeamer' package 
(e.g., "qtpy.QtBindingsNotFoundError: No Qt bindings could be found"), please try the following:

  `
  pip install manim-slides[pyside6]
  `
  
