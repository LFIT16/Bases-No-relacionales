"""
evaluacion/dataset.py
=====================
Dataset de evaluación con 20 pares (pregunta, ground_truth) derivados
directamente del corpus de reportes aeroportuarios existente.

Cada entrada incluye:
  - question    : consulta en lenguaje natural
  - ground_truth: respuesta correcta esperada, extraída del contenido de los reportes
  - tipo        : tipo de reporte fuente (para análisis por categoría)
  - prioridad   : prioridad del reporte fuente
"""

EVAL_DATASET = [
    # ── INCIDENTES ────────────────────────────────────────────────────────────
    {
        "id": "eval_001",
        "question": "¿Qué ocurrió en la pista 18L el 12 de enero de 2024?",
        "ground_truth": (
            "Un vehículo de remolque colisionó levemente con una escalerilla en la pista 18L. "
            "No hubo heridos. Se activó el protocolo de inspección de pista y se reportó al área "
            "de mantenimiento para revisión del vehículo involucrado."
        ),
        "tipo": "incidente",
        "prioridad": "alta",
    },
    {
        "id": "eval_002",
        "question": "¿Qué pasó con el carrusel de equipaje número 3?",
        "ground_truth": (
            "El carrusel número 3 de la terminal nacional presentó fallo mecánico, afectando la "
            "entrega de equipaje de 4 vuelos. Se activó el protocolo alterno con uso de carruseles "
            "4 y 5. El fallo fue reparado en 2 horas y se emitieron disculpas a los pasajeros afectados."
        ),
        "tipo": "incidente",
        "prioridad": "media",
    },
    {
        "id": "eval_003",
        "question": "¿Qué incidente ocurrió con el vuelo AV234 en marzo de 2024?",
        "ground_truth": (
            "El vuelo AV234 reportó impacto de ave en el motor número 1 durante el despegue del "
            "12 de marzo. El comandante declaró emergencia y regresó a la pista. El motor fue "
            "inspeccionado por ingeniería y la aeronave fue liberada horas después."
        ),
        "tipo": "incidente",
        "prioridad": "alta",
    },
    {
        "id": "eval_004",
        "question": "¿Cómo se manejó al pasajero con comportamiento disruptivo en el vuelo LA4521?",
        "ground_truth": (
            "Durante el vuelo LA4521 un pasajero presentó comportamiento agresivo hacia la "
            "tripulación de cabina. El piloto notificó a tierra y al aterrizar el pasajero fue "
            "entregado a la policía aeroportuaria."
        ),
        "tipo": "incidente",
        "prioridad": "alta",
    },
    {
        "id": "eval_005",
        "question": "¿Qué ocurrió con el derrame de combustible en la plataforma norte?",
        "ground_truth": (
            "Una manguera de abastecimiento presentó fuga durante la carga de combustible en la "
            "aeronave HK-5089. El derrame fue de aproximadamente 200 litros. Se activó el "
            "protocolo de contención ambiental y la plataforma fue cerrada temporalmente para "
            "limpieza y descontaminación."
        ),
        "tipo": "incidente",
        "prioridad": "alta",
    },
    # ── CLIMÁTICOS ────────────────────────────────────────────────────────────
    {
        "id": "eval_006",
        "question": "¿Por qué se suspendieron operaciones el 15 de enero de 2024?",
        "ground_truth": (
            "La tormenta eléctrica registrada el 15 de enero de 2024 obligó a suspender todas "
            "las operaciones de despegue y aterrizaje durante 3 horas. Se activaron protocolos "
            "de seguridad meteorológica. Los pasajeros fueron reubicados en salas de espera y "
            "los vuelos reprogramados."
        ),
        "tipo": "climatico",
        "prioridad": "alta",
    },
    {
        "id": "eval_007",
        "question": "¿Qué procedimiento se aplicó cuando hubo niebla densa con visibilidad menor a 200 metros?",
        "ground_truth": (
            "Cuando se registró niebla densa con visibilidad menor a 200 metros en las pistas "
            "13R y 13L, se aplicó el procedimiento CAT III de aproximación por instrumentos. "
            "Las operaciones se redujeron a una pista y con intervalos de seguridad ampliados."
        ),
        "tipo": "climatico",
        "prioridad": "alta",
    },
    {
        "id": "eval_008",
        "question": "¿Qué daños causó el granizo del 18 de marzo en las aeronaves?",
        "ground_truth": (
            "El fuerte granizo del 18 de marzo causó abolladuras en 6 aeronaves estacionadas en "
            "plataforma abierta. Las aeronaves afectadas pertenecían a Latam, Avianca y Wingo. "
            "Todas fueron sometidas a inspección técnica antes de retornar a operaciones."
        ),
        "tipo": "climatico",
        "prioridad": "alta",
    },
    {
        "id": "eval_009",
        "question": "¿Qué pasó durante los vientos cruzados de 45 nudos del 28 de febrero?",
        "ground_truth": (
            "Los vientos cruzados de 45 nudos impidieron operaciones en la pista principal. "
            "Solo la pista 04 estuvo operativa. Varios vuelos fueron desviados a aeropuertos "
            "alternos y las operaciones nocturnas fueron suspendidas."
        ),
        "tipo": "climatico",
        "prioridad": "alta",
    },
    # ── SEGURIDAD ─────────────────────────────────────────────────────────────
    {
        "id": "eval_010",
        "question": "¿Qué encontró la unidad canina en la sala de embarque B7?",
        "ground_truth": (
            "La unidad canina detectó un artículo sospechoso en la sala de embarque B7. Se evacuó "
            "el área preventivamente. Tras inspección exhaustiva, el artículo resultó ser material "
            "electrónico olvidado por un pasajero. Las operaciones se reanudaron en 45 minutos."
        ),
        "tipo": "seguridad",
        "prioridad": "alta",
    },
    {
        "id": "eval_011",
        "question": "¿Cuántas personas participaron en el simulacro de evacuación del 18 de febrero?",
        "ground_truth": (
            "En el simulacro de evacuación total del terminal principal del 18 de febrero "
            "participaron 450 empleados. Se simuló un incendio en el área de facturación y "
            "se evaluó el tiempo de respuesta de los equipos de emergencia."
        ),
        "tipo": "seguridad",
        "prioridad": "media",
    },
    {
        "id": "eval_012",
        "question": "¿Qué incautó la DIAN en el vuelo con destino a Panamá el 5 de marzo?",
        "ground_truth": (
            "La DIAN junto con seguridad aeroportuaria incautó 380,000 dólares no declarados "
            "en un vuelo con destino a Panamá. Los billetes estaban distribuidos en el equipaje "
            "de mano y documentado de dos pasajeros."
        ),
        "tipo": "seguridad",
        "prioridad": "alta",
    },
    {
        "id": "eval_013",
        "question": "¿Para qué sirven los nuevos escáneres de tomografía computarizada CT instalados en marzo?",
        "ground_truth": (
            "Los escáneres de tomografía computarizada CT instalados en las líneas de seguridad "
            "internacionales permiten visualizar el contenido del equipaje en 3D sin necesidad "
            "de abrir las maletas, reduciendo los tiempos de inspección y mejorando la detección "
            "de objetos prohibidos."
        ),
        "tipo": "seguridad",
        "prioridad": "media",
    },
    # ── MANTENIMIENTO ─────────────────────────────────────────────────────────
    {
        "id": "eval_014",
        "question": "¿Qué componentes se inspeccionaron en el mantenimiento del Boeing 737-800 HK-4321?",
        "ground_truth": (
            "En el mantenimiento programado de la aeronave HK-4321 (Boeing 737-800) se "
            "inspeccionaron motores CFM56, sistema hidráulico y tren de aterrizaje. Todas "
            "las verificaciones resultaron exitosas y la aeronave fue liberada para operaciones "
            "normales al día siguiente."
        ),
        "tipo": "mantenimiento",
        "prioridad": "media",
    },
    {
        "id": "eval_015",
        "question": "¿Qué problema se reportó con las balizas de la pista 31 el 7 de febrero?",
        "ground_truth": (
            "Se reportó el fallo de tres balizas de aproximación en la pista 31. El equipo de "
            "mantenimiento eléctrico realizó reparación de emergencia y las balizas quedaron "
            "operativas antes del inicio del tráfico nocturno."
        ),
        "tipo": "mantenimiento",
        "prioridad": "alta",
    },
    {
        "id": "eval_016",
        "question": "¿En qué consiste la actualización del sistema ILS de la pista 13L realizada en febrero?",
        "ground_truth": (
            "La actualización del sistema ILS (Instrument Landing System) de la pista 13L "
            "se completó el 21 de febrero. El nuevo sistema permite aproximaciones de categoría "
            "CAT IIIb con visibilidad de hasta 50 metros, mejorando significativamente la "
            "capacidad operativa en condiciones meteorológicas adversas."
        ),
        "tipo": "mantenimiento",
        "prioridad": "media",
    },
    # ── OPERATIVOS ────────────────────────────────────────────────────────────
    {
        "id": "eval_017",
        "question": "¿Cuáles son las características de la nueva ruta Bogotá-Miami inaugurada por Avianca?",
        "ground_truth": (
            "Avianca inauguró el 20 de enero de 2024 la ruta directa Bogotá-Miami con "
            "frecuencia diaria. El vuelo AV045 opera en un Airbus A330-200 con duración "
            "aproximada de 4 horas y 20 minutos, con un promedio esperado de 180 pasajeros "
            "por vuelo."
        ),
        "tipo": "operativo",
        "prioridad": "baja",
    },
    {
        "id": "eval_018",
        "question": "¿Cómo funciona el sistema biométrico de embarque implementado desde el 1 de febrero?",
        "ground_truth": (
            "Desde el 1 de febrero el aeropuerto implementó reconocimiento facial para embarque "
            "en puertas internacionales A1-A10. El sistema procesa a cada pasajero en menos de "
            "3 segundos comparando la imagen en tiempo real con la base de datos de documentos "
            "registrados al momento del check-in."
        ),
        "tipo": "operativo",
        "prioridad": "baja",
    },
    {
        "id": "eval_019",
        "question": "¿Cuántas pantallas tiene el nuevo sistema FIDS activado en marzo?",
        "ground_truth": (
            "El nuevo sistema FIDS (Flight Information Display System) activado el 15 de marzo "
            "cuenta con 320 pantallas 4K distribuidas en todo el aeropuerto, mostrando "
            "información de vuelos en tiempo real en español e inglés."
        ),
        "tipo": "operativo",
        "prioridad": "baja",
    },
    {
        "id": "eval_020",
        "question": "¿Cuál es la capacidad de la nueva terminal de carga inaugurada el 1 de marzo?",
        "ground_truth": (
            "La nueva terminal de carga inaugurada el 1 de marzo de 2024 tiene capacidad para "
            "15,000 toneladas mensuales, el doble de la terminal anterior. Cuenta con sistemas "
            "de refrigeración para carga perecedera y certificación CEIV Pharma para "
            "medicamentos y vacunas."
        ),
        "tipo": "operativo",
        "prioridad": "baja",
    },
]
