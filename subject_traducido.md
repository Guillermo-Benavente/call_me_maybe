
# Call Me Maybe
## Introducción al Function Calling en LLMs

**Resumen:** ¿Los LLM hablan el lenguaje de los ordenadores? Vamos a descubrirlo.

Realizado en colaboración con @ldevelle, @pcamaren, @crfernan.

**Versión: 1.5**

---

# Capítulo I — Prólogo

Los ingenieros romanos grababan tablas de piedra con cuadrículas perfectas para llevar un registro de los envíos de grano por todo el imperio, de modo que ninguna entrega se perdiera por una mala caligrafía. En el siglo XIX, los marineros registraban las corrientes oceánicas en tablas estructuradas tan precisas que algunas todavía se utilizan hoy en navegación.

Las primeras predicciones meteorológicas se enviaban mediante telegramas utilizando un código fijo: unos pocos números podían describir todo un cielo. En la década de 1960, el control de misiones de la NASA trabajaba con diagramas de flujo plastificados que indicaban exactamente qué significaba cada luz parpadeante, independientemente de quién estuviera de turno.

Los lectores de códigos de barras de los supermercados traducían las franjas blancas y negras en datos de inventario mucho antes de que la mayoría de los hogares tuvieran ordenadores. Incluso los apicultores llevan décadas utilizando formularios estandarizados para registrar la salud de las colmenas, el flujo de néctar y el linaje de las reinas en pequeñas casillas que podían interpretar de un vistazo.

Los seres humanos siempre han construido estructuras para hacer que la información sea fiable, compartible y utilizable.

Y eso nos lleva hasta hoy, donde el objetivo es conseguir que la IA hable el lenguaje de los ordenadores.

---

# Capítulo II — Instrucciones sobre IA

## Contexto

Durante tu proceso de aprendizaje, la IA puede ayudarte con muchas tareas diferentes. Tómate el tiempo necesario para explorar las distintas capacidades de las herramientas de IA y cómo pueden ayudarte en tu trabajo.

Sin embargo, utilízalas siempre con precaución y evalúa críticamente los resultados. Ya sea código, documentación, ideas o explicaciones técnicas, nunca puedes estar completamente seguro de que tu pregunta haya sido formulada correctamente o de que el contenido generado sea preciso.

Tus compañeros son un recurso muy valioso para ayudarte a evitar errores y puntos ciegos.

## Mensaje principal

- Utiliza la IA para reducir tareas repetitivas o tediosas.
- Desarrolla habilidades de prompting, tanto relacionadas con código como no relacionadas con él, que te serán útiles en tu futura carrera.
- Aprende cómo funcionan los sistemas de IA para anticipar y evitar mejor riesgos, sesgos y problemas éticos comunes.
- Continúa desarrollando tus habilidades técnicas y profesionales trabajando con tus compañeros.
- Utiliza únicamente contenido generado por IA que comprendas completamente y del que puedas responsabilizarte.

## Reglas para el estudiante

- Debes dedicar tiempo a explorar las herramientas de IA y comprender cómo funcionan para poder utilizarlas de forma ética y reducir posibles sesgos.
- Debes reflexionar sobre tu problema antes de escribir el prompt. Esto te ayudará a escribir prompts más claros, detallados y relevantes utilizando un vocabulario preciso.
- Debes desarrollar el hábito de comprobar, revisar, cuestionar y probar sistemáticamente todo aquello que genere la IA.
- Debes buscar siempre una revisión por parte de compañeros. No dependas únicamente de tu propia validación.

## Resultados de la fase

- Desarrollar habilidades de prompting tanto generales como específicas del dominio.
- Aumentar tu productividad mediante un uso eficaz de las herramientas de IA.
- Seguir fortaleciendo el pensamiento computacional, la resolución de problemas, la adaptabilidad y la colaboración.

## Comentarios y ejemplos

- Te encontrarás regularmente con situaciones —exámenes, evaluaciones y otras— en las que tendrás que demostrar que realmente comprendes lo que haces. Prepárate y continúa desarrollando tanto tus habilidades técnicas como interpersonales.
- Explicar tu razonamiento y debatir con tus compañeros suele revelar lagunas en tu comprensión. Haz del aprendizaje entre compañeros una prioridad.
- Las herramientas de IA suelen carecer de tu contexto específico y tienden a proporcionar respuestas genéricas. Tus compañeros, que comparten tu entorno, pueden ofrecer información más relevante y precisa.
- Mientras que la IA tiende a generar la respuesta más probable, tus compañeros pueden aportar perspectivas alternativas y matices importantes. Utilízalos como punto de control de calidad.

### ✓ Buena práctica

Le pregunto a la IA:

> «¿Cómo pruebo una función de ordenación?»

Me da algunas ideas. Las pruebo y reviso los resultados con un compañero. Juntos perfeccionamos el enfoque.

### ✗ Mala práctica

Le pido a la IA que escriba una función completa y la copio y pego en mi proyecto. Durante la evaluación entre compañeros, no puedo explicar qué hace ni por qué. Pierdo credibilidad y suspendo el proyecto.

### ✓ Buena práctica

Utilizo la IA para ayudarme a diseñar un parser. Después reviso la lógica con un compañero. Detectamos dos errores y lo reescribimos juntos: mejor, más limpio y completamente comprendido.

### ✗ Mala práctica

Dejo que Copilot genere el código de una parte importante de mi proyecto. Compila, pero no puedo explicar cómo gestiona los pipes. Durante la evaluación, no puedo justificarlo y suspendo el proyecto.

---

# Capítulo III — Introducción

## III.1 ¿Qué es Function Calling?

Los Large Language Models (LLM) son muy potentes a la hora de comprender y generar lenguaje humano, pero no producen de forma natural una salida estructurada y ejecutable por una máquina.

El **function calling** cierra esta brecha traduciendo solicitudes expresadas en lenguaje natural en llamadas precisas a funciones con argumentos tipados.

Por ejemplo:

### Lenguaje natural → Function Call

Usuario:

```text
"What is the sum of 40 and 2?"
```

LLM tradicional:

```text
"The sum of 40 and 2 is 42."
```

Sistema de Function Calling:

```json
{
    "function": "add_numbers",
    "arguments": {
        "a": 40,
        "b": 2
    }
}
```

El sistema de function calling no responde directamente a la pregunta.

En su lugar, proporciona las herramientas necesarias para resolverla:

- el nombre correcto de la función;
- los argumentos correctos;
- los tipos correctos de esos argumentos.

## III.2 ¿Por qué es importante?

El function calling permite a los LLM:

- **Interactuar con sistemas externos:** llamar a APIs, consultar bases de datos y controlar dispositivos.
- **Ejecutar código:** realizar cálculos, transformaciones de datos y operaciones con archivos.
- **Encadenar operaciones:** dividir tareas complejas en pasos ejecutables.
- **Proporcionar salidas estructuradas:** generar JSON, XML u otros formatos legibles por máquinas.
- **Extraer datos estructurados de texto no estructurado:** por ejemplo, dado un libro grande, extraer campos como:

```text
{nombre del protagonista, sexo del protagonista, edad del protagonista}
```

Esta tecnología impulsa los asistentes modernos de IA, las herramientas de generación de código y los agentes autónomos. También permite tareas como la extracción automática de información y la estructuración del conocimiento a partir de texto sin procesar.

## III.3 El desafío

Los modelos de lenguaje pequeños, como el modelo de 0.6B parámetros que utilizarás, son conocidos por ser poco fiables a la hora de generar salidas estructuradas.

Cuando se les pide que produzcan JSON, podrían hacerlo correctamente solamente alrededor del 30 % de las veces.

Sin embargo, los sistemas en producción consiguen una fiabilidad superior al 99 % utilizando estos mismos modelos pequeños.

¿Cómo?

La respuesta está en el **constrained decoding**: una técnica que guía la salida del modelo token a token para garantizar una estructura válida, sin depender únicamente del prompting.

---

# Capítulo IV — Instrucciones comunes

## IV.1 Reglas generales

- El proyecto debe estar escrito en **Python 3.10 o superior**.
- El proyecto debe cumplir el estándar de código **flake8**.
- Tus funciones deben gestionar las excepciones correctamente para evitar crashes. Utiliza bloques `try-except` para gestionar posibles errores.
- Se recomienda utilizar context managers para recursos como archivos o conexiones, garantizando su limpieza automática.
- Si el programa se bloquea debido a excepciones no gestionadas durante la revisión, se considerará que no funciona.
- Todos los recursos, como manejadores de archivos o conexiones de red, deben gestionarse correctamente para evitar fugas.
- Cuando sea posible, utiliza context managers para gestionar automáticamente estos recursos.
- El código debe incluir **type hints** para los parámetros, valores de retorno y variables cuando corresponda, utilizando el módulo `typing`.
- Debes utilizar **mypy** para realizar comprobaciones estáticas.
- Todas las funciones deben pasar `mypy` sin errores.
- Incluye **docstrings** en funciones y clases siguiendo PEP 257, por ejemplo utilizando el estilo Google o NumPy, documentando propósito, parámetros y valores de retorno.

## IV.2 Makefile

Debes incluir un `Makefile` en el proyecto para automatizar las tareas habituales.

Debe contener las siguientes reglas:

### `install`

Instala las dependencias del proyecto utilizando `pip`, `uv`, `pipx` o cualquier otro gestor de paquetes que elijas.

### `run`

Ejecuta el script principal del proyecto, por ejemplo mediante el intérprete de Python.

### `debug`

Ejecuta el script principal en modo debug utilizando el depurador incluido en Python, por ejemplo `pdb`.

### `clean`

Elimina archivos temporales o cachés, como:

```text
__pycache__
.mypy_cache
```

para mantener limpio el entorno del proyecto.

### `lint`

Debe ejecutar:

```bash
flake8 .
mypy . --warn-return-any \
       --warn-unused-ignores \
       --ignore-missing-imports \
       --disallow-untyped-defs \
       --check-untyped-defs
```

### `lint-strict` (opcional)

Debe ejecutar:

```bash
flake8 .
mypy . --strict
```

## IV.3 Directrices adicionales

- Crea programas de prueba para verificar el funcionamiento del proyecto. Estos no se entregan ni se califican.
- Utiliza frameworks como `pytest` o `unittest` para realizar pruebas unitarias, cubriendo casos límite.
- Incluye un archivo `.gitignore` para excluir archivos propios de Python.
- Se recomienda utilizar entornos virtuales, como `venv` o `conda`, para aislar las dependencias durante el desarrollo.

Si existen requisitos adicionales específicos del proyecto, aparecerán inmediatamente después de esta sección.

## IV.3.1 Requisitos adicionales

- Todas las clases deben utilizar **pydantic** para la validación.
- Puedes utilizar los paquetes `numpy` y `json`.
- Está completamente prohibido utilizar `dspy` o cualquier paquete similar, incluyendo:
  - `pytorch`
  - `huggingface`
  - `transformers`
  - `outlines`
  - etc.
- Debes utilizar los siguientes modelos:

```text
Qwen/Qwen3-0.6B
```

Este es el modelo predeterminado.

Puedes utilizar otros modelos siempre que tu proyecto funcione con `Qwen/Qwen3-0.6B`.

- La función que se debe llamar debe ser elegida utilizando el LLM, no mediante heurísticas ni ningún otro tipo de «magia medieval».
- Está prohibido utilizar métodos o atributos privados del paquete `llm_sdk`.
- Debes crear un entorno virtual e instalar los paquetes `numpy` y `pydantic` utilizando `uv`.
- Para utilizar `llm_sdk`, puedes copiarlo en el mismo directorio donde se encuentra `src`.
- El reviewer y la moulinette simplemente ejecutarán:

```bash
uv sync
```

- Todos los errores deben gestionarse correctamente.
- El programa nunca debe bloquearse inesperadamente y siempre debe proporcionar mensajes de error claros al usuario.

## IV.3.2 Uso

Tu programa debe ejecutarse utilizando el siguiente comando, donde `src` es la carpeta que contiene tus archivos:

```bash
uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]
```

Por defecto, el programa leerá los archivos de entrada desde:

```text
data/input/
```

y escribirá los resultados en:

```text
data/output/
```

Puedes especificar rutas personalizadas mediante `--input` y `--output`.

Por ejemplo:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

---

# Capítulo V — Parte obligatoria

## V.1 Resumen

En este proyecto crearás una herramienta de **function calling** que traduzca prompts escritos en lenguaje natural en llamadas estructuradas a funciones.

Dada una pregunta como:

```text
"What is the sum of 40 and 2?"
```

tu solución **no debe devolver**:

```text
42
```

En su lugar debe proporcionar:

- Nombre de la función:

```text
fn_add_numbers
```

- Argumentos:

```json
{
    "a": 40,
    "b": 2
}
```

Tu implementación debe utilizar **constrained decoding** para garantizar una salida JSON 100 % válida, consiguiendo una fiabilidad prácticamente perfecta incluso utilizando un modelo pequeño de 0.6B parámetros.

---

# V.2 Archivos de entrada

Tu solución procesará dos archivos situados en:

```text
data/input/
```

### `function_calling_tests.json`

Contiene un array JSON de prompts escritos en lenguaje natural que tu sistema debe procesar.

Ejemplo:

```json
[
    {
        "prompt": "What is the sum of 2 and 3?"
    },
    {
        "prompt": "What is the sum of 265 and 345?"
    },
    {
        "prompt": "Greet shrek"
    },
    {
        "prompt": "Greet john"
    },
    {
        "prompt": "Reverse the string 'hello'"
    }
]
```

### `functions_definition.json`

Contiene las funciones disponibles que tu sistema puede llamar.

Cada función incluye:

- Nombre de la función.
- Nombres y tipos de los argumentos.
- Tipo de retorno.
- Descripción.

Ejemplo:

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    },
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {
            "name": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    },
    {
        "name": "fn_reverse_string",
        "description": "Reverse a string and return the reversed result.",
        "parameters": {
            "s": {
                "type": "string"
            }
        },
        "returns": {
            "type": "string"
        }
    }
]
```

Estos ejemplos establecen el nivel de complejidad esperado.

Sin embargo, tu solución será probada con diferentes prompts y diferentes conjuntos de funciones.

Debes implementar un manejo adecuado de errores JSON para los archivos de entrada, ya que pueden contener JSON inválido o incluso no existir.

---

# V.3 Interacción con el LLM

## V.3.1 El SDK del LLM

Junto con este proyecto encontrarás una clase wrapper llamada:

```text
Small_LLM_Model
```

dentro del paquete:

```text
llm_sdk
```

que puedes utilizar para interactuar con el LLM.

El SDK proporciona varios métodos esenciales.

### `get_logits_from_input_ids`

```python
get_logits_from_input_ids(input_ids: List[int]) -> List[float]
```

Recibe una lista de IDs de tokens y devuelve los logits producidos por el modelo.

### `get_path_to_vocab_file`

```python
get_path_to_vocab_file() -> str
```

Devuelve la ruta al archivo de vocabulario que contiene la correspondencia entre IDs de tokens y tokens.

### `encode`

```python
encode(text: str) -> Tensor
```

Codifica un texto en un tensor de IDs de tokens utilizando el tokenizer del modelo.

### `decode`

```python
decode(token_ids: List[int]) -> str
```

Método opcional que decodifica una lista de IDs de tokens de nuevo a texto.

---

# V.3.2 Pipeline de generación

El proceso de generación del LLM sigue estos pasos:

### 1. Prompt

Tu pregunta en lenguaje natural.

Ejemplo:

```text
"What is the sum of 2 and 3?"
```

### 2. Tokenización

El texto se divide en unidades más pequeñas llamadas **tokens**.

A diferencia de una simple separación por palabras, los tokenizers suelen:

- incluir espacios iniciales;
- gestionar signos de puntuación;
- dividir palabras en componentes más pequeños.

Utilizan algoritmos como BPE o SentencePiece.

Ejemplo realista:

```text
["What", "Gis ˙ ", "Gthe ˙ ", "Gsum ˙ ", "Gof ˙ ", "G2˙ ", "Gand ˙ ", "G3˙ ", "?"]
```

El símbolo `˙G` indica un espacio precedente.

Los tokenizers reales conservan estos detalles para poder reconstruir el texto correctamente.

### 3. Input IDs

Los tokens se convierten en IDs numéricos que el modelo puede entender.

Ejemplo:

```text
[892, 318, 262, 4771, 286, 16, 290, 17, 30]
```

### 4. Procesamiento del LLM

El modelo procesa esos números mediante su red neuronal.

### 5. Logits

El modelo genera puntuaciones de probabilidad para cada posible token siguiente.

Ejemplo:

```text
token_5:   0.001
token_42:  0.85
token_100: 0.02
...
```

### 6. Selección del token

Se selecciona el siguiente token basándose en estas probabilidades. Normalmente se selecciona el que tiene la puntuación más alta.

En este punto se pueden aplicar técnicas como **constrained decoding** para limitar los tokens disponibles y garantizar que la salida siga una estructura específica, como un JSON 100 % válido.

Este proceso se repite **token a token**.

Cada token generado se añade al prompt y los pasos 2-6 vuelven a repetirse hasta generar la respuesta completa.

### Vista simplificada

```text
Prompt
   ↓
Tokenization
   ↓
Input IDs
   ↓
LLM
   ↓
Logits
   ↓
Next Token Selection
```

---

# V.3.3 Entendiendo Constrained Decoding

Los modelos de lenguaje generan texto un token cada vez.

En cada paso, el modelo produce una distribución de probabilidades —logits— sobre todos los posibles tokens.

Normalmente puedes muestrear a partir de esta distribución o seleccionar el token con mayor probabilidad.

El **constrained decoding** interviene en este proceso modificando los logits antes de seleccionar el token.

### Proceso

1. El modelo genera logits para todos los tokens posibles.
2. Identificas qué tokens mantendrían simultáneamente:
   - una estructura JSON válida;
   - el cumplimiento del esquema esperado.
3. Estableces los logits de los tokens inválidos —aquellos que romperían el esquema o la estructura— en infinito negativo.
4. Seleccionas únicamente entre los tokens válidos restantes.

En este proyecto, el constrained decoding no solamente debe garantizar que el JSON sea sintácticamente válido, sino que también debe imponer un esquema específico.

Por ejemplo, si un campo está restringido al tipo `number` en `functions_definition.json`, el decoder debe limitar la selección de tokens a valores que sean un entero o un número decimal.

Esto mantiene tanto:

- la validez del JSON;
- como el cumplimiento del esquema.

Como resultado, cada token generado mantiene tanto la validez estructural como la semántica requerida.

El resultado es un JSON 100 % válido que puede analizarse siempre sin errores.

### Importante

Tu solución **NO debe depender de que el modelo genere espontáneamente un JSON correcto**.

No es suficiente con proporcionar al modelo las definiciones de las funciones mediante un prompt y esperar que genere una salida estructurada correctamente.

Eso no es fiable y **no es la habilidad que se espera que desarrolles en este proyecto**.

Piensa en cómo puedes utilizar el archivo JSON del vocabulario para establecer una correspondencia entre:

```text
tokens ↔ strings
```

Esto es fundamental para determinar qué tokens son válidos en cada paso de generación.

---

# V.4 Formato del archivo de salida

Tu programa producirá un único archivo JSON:

```text
data/output/function_calling_results.json
```

Por cada prompt debes añadir un objeto JSON a este archivo.

Cada objeto del array debe contener **exactamente** las siguientes claves:

### `prompt`

Tipo:

```text
string
```

La solicitud original en lenguaje natural.

### `name`

Tipo:

```text
string
```

El nombre de la función que debe llamarse.

### `parameters`

Tipo:

```text
object
```

Todos los argumentos requeridos con los tipos correctos.

---

# V.4.1 Ejemplo de salida

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    },
    {
        "prompt": "Reverse the string 'hello'",
        "name": "fn_reverse_string",
        "parameters": {
            "s": "hello"
        }
    }
]
```

---

# V.4.2 Reglas de validación

- El archivo debe ser JSON válido.
- No puede haber comas finales.
- No puede haber comentarios.
- Las claves y los tipos deben coincidir exactamente con el esquema de `functions_definition.json`.
- No se permiten claves adicionales ni texto explicativo en ninguna parte de la salida.
- Todos los argumentos requeridos deben estar presentes.
- Los tipos de los argumentos deben coincidir con la definición de la función:
  - `number`
  - `string`
  - `boolean`
  - etc.

Los archivos de entrada proporcionados pueden cambiar durante la evaluación entre compañeros.

**No hardcodees soluciones basadas en los ejemplos proporcionados.**

---

# V.5 Rendimiento y fiabilidad

Tu implementación debería conseguir:

- **Precisión casi perfecta:** más del 90 % de selección correcta de funciones y extracción correcta de argumentos.
- **JSON 100 % válido:** cada salida debe poder analizarse y cumplir el esquema.
- **Velocidad razonable:** procesar todos los prompts de prueba en menos de 5 minutos utilizando hardware estándar.
- **Gestión robusta de errores:** gestionar correctamente entradas malformadas, archivos inexistentes y casos límite.

El modelo `Qwen3-0.6B` solamente tiene 500 millones de parámetros, pero con un constrained decoding adecuado puede conseguir una fiabilidad comparable a modelos mucho más grandes.

Esto demuestra el poder de la **guía estructural frente al tamaño bruto del modelo**.

---

# V.6 Probando tu implementación

Para verificar que funciona correctamente:

1. Asegúrate de que los archivos de entrada están en:

```text
data/input/
```

2. Ejecuta:

```bash
uv run python -m src \
    [--functions_definition <function_definition_file>] \
    [--input <input_file>] \
    [--output <output_file>]
```

3. Comprueba que se ha creado:

```text
output/function_calling_results.json
```

4. Valida la estructura y el contenido del JSON.

5. Comprueba que los nombres de las funciones y los tipos de los argumentos coinciden con las definiciones.

Prueba diferentes casos límite:

- strings vacíos;
- números grandes;
- caracteres especiales;
- tipos incorrectos;
- prompts ambiguos;
- funciones con múltiples parámetros.

---

# Capítulo VI — Requisitos del README

Debe existir un archivo:

```text
README.md
```

en la raíz de tu repositorio Git.

Su objetivo es permitir que cualquier persona que no conozca el proyecto —compañeros, staff, recruiters, etc.— pueda entender rápidamente:

- de qué trata el proyecto;
- cómo ejecutarlo;
- dónde encontrar más información sobre el tema.

El `README.md` debe incluir como mínimo:

### Primera línea

La primera línea debe estar en cursiva y decir:

```text
This project has been created as part of the 42 curriculum by <login1>[, <login2>[, <login3>[...]]].
```

### Description

Una sección llamada:

```text
Description
```

que explique claramente el proyecto, incluyendo su objetivo y una breve descripción general.

### Instructions

Una sección llamada:

```text
Instructions
```

que contenga toda la información relevante sobre:

- compilación;
- instalación;
- ejecución.

### Resources

Una sección llamada:

```text
Resources
```

que enumere referencias clásicas relacionadas con el tema:

- documentación;
- artículos;
- tutoriales;
- etc.

También debe explicar cómo se utilizó la IA, especificando para qué tareas y partes del proyecto.

Dependiendo del proyecto pueden ser necesarias secciones adicionales, por ejemplo:

- ejemplos de uso;
- lista de funcionalidades;
- decisiones técnicas;
- etc.

Cualquier requisito adicional se especificará explícitamente.

## Requisitos específicos del README para este proyecto

El README también debe incluir:

- **Explicación del algoritmo:** describir detalladamente tu enfoque de constrained decoding.
- **Decisiones de diseño:** explicar las decisiones importantes tomadas durante la implementación.
- **Análisis de rendimiento:** hablar sobre precisión, velocidad y fiabilidad.
- **Dificultades encontradas:** documentar los problemas encontrados y cómo fueron solucionados.
- **Estrategia de testing:** explicar cómo validaste la implementación.
- **Ejemplos de uso:** proporcionar ejemplos claros de cómo ejecutar el programa.

**El README debe estar escrito en inglés.**

---

# Capítulo VII — Parte Bonus

Comprueba las siguientes funcionalidades bonus.

Son opcionales y no son necesarias para aprobar:

- Soporte para múltiples modelos LLM además de `Qwen/Qwen3-0.6B`.
- Recodificación del tokenizer:
  - evitar el uso directo de `encode` y `decode` en el código principal;
  - utilizar en su lugar `get_logits_from_input_ids` y `get_path_to_vocab_file`.
- Mecanismos avanzados de recuperación ante errores.
- Optimizaciones de rendimiento:
  - caching;
  - batching.
- Suite completa de tests.
- Visualización del proceso de generación.
- Soporte para argumentos de funciones complejos y anidados.
- Implementación pública de los métodos `encode` y `decode` opcional del tokenizer.
- Demostración de cómo el encoding y decoding se integran con constrained decoding.

Las funcionalidades bonus deben estar **implementadas y funcionando**, no simplemente descritas en el README.

Es posible que tengas que demostrarlas durante la evaluación.

---

# Capítulo VIII — Entrega y evaluación entre compañeros

Entrega el proyecto en tu repositorio Git como de costumbre.

Durante la defensa solamente se revisará el trabajo que esté dentro de tu repositorio.

No dudes en comprobar dos veces los nombres de tus archivos para asegurarte de que son correctos.

Tu repositorio debe contener:

```text
src/
```

con tu implementación.

```text
pyproject.toml
uv.lock
```

para la gestión de dependencias.

```text
llm_sdk/
```

copiado desde el paquete proporcionado.

```text
data/input/
```

con archivos de prueba para demostrar el funcionamiento.

```text
README.md
```

con documentación completa.

Y cualquier archivo adicional necesario para ejecutar tu solución.

### No incluyas:

```text
output/
```

en el repositorio.

Esta carpeta será generada durante la evaluación entre compañeros.

---

Durante la evaluación puede solicitarse ocasionalmente una pequeña modificación del proyecto.

Esto podría consistir en:

- un pequeño cambio de comportamiento;
- escribir o reescribir unas pocas líneas de código;
- añadir una funcionalidad sencilla.

Aunque este paso puede no aplicarse a todas las evaluaciones, debes estar preparado si aparece en las instrucciones de evaluación.

El objetivo de este paso es comprobar que realmente comprendes una parte concreta del proyecto.

La modificación puede realizarse en cualquier entorno de desarrollo que elijas y debería poder completarse en unos pocos minutos, salvo que las instrucciones de evaluación indiquen otro límite de tiempo.

Por ejemplo, podrían pedirte:

- realizar una pequeña modificación en una función o script;
- modificar la visualización;
- cambiar una estructura de datos para almacenar nueva información;
- etc.

Los detalles —alcance, objetivo, etc.— se especificarán en las instrucciones de evaluación y pueden variar de una evaluación a otra para el mismo proyecto.