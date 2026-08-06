# ZeyLang

Bahasa pemrograman untuk **AI, Robot, dan Space** — syntax simpel ala Python,
dengan `ai`, `robot`, `space` sebagai namespace bawaan. Method dipanggil pakai
`@`, bukan `.`. Bisa dijalankan langsung (interpreter/REPL) atau di-compile
ke C native.

```zey
ai@chat("Halo dunia!")
robot@walk(10)
space@orbit(400, "km")
