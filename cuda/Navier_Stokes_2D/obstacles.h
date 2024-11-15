#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Macro per calcolare l'indice in base alle coordinate x, y
#define IND(x, y, d) int((y) * (d) + (x))

// Funzione per inizializzare un ostacolo cilindrico
void initializeCylinder(int* obstacleField, unsigned dim, float centerX, float centerY, float radius) {
    for (unsigned i = 0; i < dim; ++i) {
        for (unsigned j = 0; j < dim; ++j) {
            float x = i + 0.5f;
            float y = j + 0.5f;
            float dx = x - centerX;
            float dy = y - centerY;
            float distance = sqrtf(dx * dx + dy * dy);

            int idx = IND(i, j, dim);
            if (distance <= radius) {
                obstacleField[idx] = 1; // Inside the cylinder
            }
        }
    }
}

// Funzione per inizializzare un ostacolo rettangolare
void initializeRectangle(int* obstacleField, unsigned dim, float startX, float startY, float width, float height) {
    for (unsigned i = 0; i < dim; ++i) {
        for (unsigned j = 0; j < dim; ++j) {
            int idx = IND(i, j, dim);
            // Verifica se il punto è nel rettangolo e se non è già un ostacolo
            if (i >= startX && i <= startX + width && j >= startY && j <= startY + height && obstacleField[idx] == 0) {
                obstacleField[idx] = 1; // Inside the rectangle (building)
            }
        }
    }
}

// Funzione per inizializzare un corridoio (strada)
void initializeCorridor(int* obstacleField, unsigned dim, float startX, float startY, float corridorWidth, bool isVertical) {
    for (unsigned i = 0; i < dim; ++i) {
        for (unsigned j = 0; j < dim; ++j) {
            int idx = IND(i, j, dim);
            // Verifica se è un corridoio e se non è già un ostacolo
            if (isVertical) {
                // Corridoio verticale
                if (i >= startX && i <= startX + corridorWidth && obstacleField[idx] == 0) {
                    obstacleField[idx] = 0; // Free space in the corridor
                }
            } else {
                // Corridoio orizzontale
                if (j >= startY && j <= startY + corridorWidth && obstacleField[idx] == 0) {
                    obstacleField[idx] = 0; // Free space in the corridor
                }
            }
        }
    }
}

// Funzione principale per inizializzare vari tipi di ostacoli
void initializeObstacle(int* obstacleField, unsigned dim, float centerX, float centerY, float radius) {
    // Imposta tutto come spazio libero all'inizio
    for (unsigned i = 0; i < dim * dim; ++i) {
        obstacleField[i] = 0;
    }

    // Crea un edificio rettangolare

    // argomenti della funzione: 
    // initializeRectangle(obstacleField, dim, startX, startY, width, height);

    // initializeRectangle(obstacleField, dim, centerX-radius-120,100, 40, 7);
    // initializeRectangle(obstacleField, dim, centerX,centerY-200, 150, 170);
    // // // // Crea un corridoio orizzontale (strada)
    // // // initializeCorridor(obstacleField, dim, 10, 100, 1000, false);

    // // // // Crea un corridoio verticale (strada)
    // // initializeCorridor(obstacleField, dim, 100, 10, 10, true);

    // // // Crea un ostacolo cilindrico
    initializeCylinder(obstacleField, dim, centerX, centerY-200, radius);
}
