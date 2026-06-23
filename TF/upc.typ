#let reporte-upc(
  curso: "",
  codigo-curso: "",
  titulo: "",
  seccion: "",
  profesor: "",
  alumnos: (),
  fecha: "",
  body,
) = {
  // 1. Configuración global del documento
  set document(title: titulo)
  set page(paper: "a4", margin: 2.5cm, numbering: "1")
  set text(font: "Liberation Serif", size: 12pt, lang: "es")
  set par(justify: true, leading: 0.65em)
  set heading(numbering: "1.1.")

  // Ajustes de espaciado ("respiros" del documento)
  set block(spacing: 1.5em)
  show heading: set block(above: 2em, below: 1em)

  // 2. Diseño de la Portada
  align(center)[
    #text(size: 16pt, weight: "bold")[Universidad Peruana de Ciencias Aplicadas]
    #v(1em)

    #box(image("media/image1.png", width: 30%))
    #v(2em)

    #text(size: 14pt)[Ciencias de la Computación] \
    #v(0.5em)
    #text(size: 12pt)[#codigo-curso - #curso] \
    #v(2em)

    #text(size: 16pt, weight: "bold")[#titulo] \
    #v(2em)

    #text(size: 12pt)[*Sección:* #seccion] \
    #v(1em)

    #text(size: 12pt)[*Profesor:*] \
    #text(size: 12pt)[#profesor] \
    #v(3em)

    // Tabla de alumnos
    #align(center)[
      #table(
        columns: (auto, auto),
        align: left,
        [*Código de alumno:*], [*Nombres y apellidos:*],
        ..alumnos.flatten(),
      )
    ]

    #v(1fr) // Empuja la fecha hacia abajo
    #text(size: 12pt)[Lima, #fecha]
  ]

  pagebreak()

  // 3. Índice automático
  outline(title: "Contenido", indent: auto)
  pagebreak()

  // 4. Inserción del contenido principal
  body
}
