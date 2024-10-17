#include <windows.h>
#include <cstdio>
#include "GameMap.h"
#include "Building.h"
#include <cstdlib>

int (*GameMap::Resource)[WIDTH] = nullptr;
int (*GameMap::Buildingsmap)[WIDTH] = nullptr;
// int (*GameMap::build)[WIDTH] = nullptr;

void GameMap::CreateMapFile(){
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,                    // Default security attributes
        PAGE_READWRITE,          // Read/write access
        0,                       // High-order file size
        sizeof(int)*HEIGHT*WIDTH, // Low-order file size (size of int)
        L"SharedResource"         // Use a wide character string for the shared memory name
        );
    Resource = (int(*)[WIDTH]) MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(int) * HEIGHT * WIDTH);

    if (Resource == NULL) {
        printf("Could not map view of file (%d).\n", GetLastError());
        CloseHandle(hMapFile);
    }
    HANDLE buildmapfile = CreateFileMapping(
        INVALID_HANDLE_VALUE,
        NULL,
        PAGE_READWRITE,
        0,
        sizeof(int)*HEIGHT*WIDTH,
        L"SharedBuild"
        );
    // build = (int(*)[WIDTH]) MapViewOfFile(buildmapfile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(int) * HEIGHT * WIDTH);
    // for (int i = 0; i < HEIGHT; i++) {

    // }
    Buildingsmap = new int[HEIGHT][WIDTH];
    HANDLE hFileMapping = CreateFileMapping(
        INVALID_HANDLE_VALUE,   // No actual file used
        NULL,                   // Default security attributes
        PAGE_READWRITE,         // Read/write access
        0,                      // High-order file size
        HEIGHT * WIDTH * sizeof(int), // Size of the mapping (int array of size 3x3)
        L"SharedBuild"             // No need to specify object name (anonymous)
        );
    LPVOID lpBase = MapViewOfFile(
        hFileMapping,           // File mapping object handle
        FILE_MAP_ALL_ACCESS,    // Read/write access
        0,                      // High-order offset
        0,                      // Low-order offset
        HEIGHT * WIDTH * sizeof(int) // Number of bytes to map
        );

    // if (lpBase == NULL) {
    //     std::cerr << "Could not map view of file. Error: " << GetLastError() << std::endl;
    //     CloseHandle(hFileMapping);
    //     return 1;
    // }

    // 3. Access the mapped area as a 2D array
    build = static_cast<int*>(lpBase); // Cast the mapped memory area to an int pointer
}

GameMap::GameMap()
{
    CreateMapFile();
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
        {
            Resource[i][j] = NONE;  // Initialize the Resource array with NONE
        }
    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
        {
            BuildingsMap[i][j] = nullptr;  // Initialize the BuildingsMap array with nullptr
            Buildingsmap[i][j] = -1;       // Initialize the Buildingsmap array with -1
            build[i * WIDTH + j] = -1;     // Initialize the build array with -1
        }
}

int* GameMap::getResource() {
    return &Resource[0][0];  // Return the starting address of the Resource array
}




void GameMap::FirstMap()
{
    Resource[1][3] = CYCLE;
    Resource[1][4] = CYCLE;
    Resource[1][5] = CYCLE;
    Resource[1][6] = CYCLE;
    Resource[2][3] = CYCLE;
    Resource[3][3] = CYCLE;
    Resource[3][4] = CYCLE;
    Resource[3][5] = CYCLE;
    Resource[3][6] = CYCLE;
    Resource[4][6] = CYCLE;
    Resource[5][6] = CYCLE;
    Resource[5][3] = CYCLE;
    Resource[5][4] = CYCLE;
    Resource[5][5] = CYCLE;

    // RECT resources
    Resource[1][25] = RECT;
    Resource[1][28] = RECT;
    Resource[2][25] = RECT;
    Resource[2][28] = RECT;
    Resource[3][25] = RECT;
    Resource[3][28] = RECT;
    Resource[4][25] = RECT;
    Resource[4][28] = RECT;
    Resource[5][25] = RECT;
    Resource[5][28] = RECT;
    Resource[3][26] = RECT;
    Resource[3][27] = RECT;

    Resource[14][3] = CYCLE;
    Resource[14][4] = CYCLE;
    Resource[14][5] = CYCLE;
    Resource[14][6] = CYCLE;
    Resource[15][3] = CYCLE;
    Resource[15][6] = CYCLE;
    Resource[16][3] = CYCLE;
    Resource[16][4] = CYCLE;
    Resource[16][5] = CYCLE;
    Resource[16][6] = CYCLE;
    Resource[17][3] = CYCLE;
    Resource[17][6] = CYCLE;
    Resource[17][6] = CYCLE;
    Resource[18][3] = CYCLE;
    Resource[18][6] = CYCLE;
    Resource[14][25] = RECT;
    Resource[14][26] = RECT;
    Resource[14][27] = RECT;
    Resource[14][28] = RECT;
    Resource[15][25] = RECT;
    Resource[15][28] = RECT;
    Resource[16][25] = RECT;
    Resource[16][28] = RECT;
    Resource[17][25] = RECT;
    //Resource[17][28] = RECT;
    Resource[16][26] = RECT;
    Resource[16][27] = RECT;
    Resource[18][25] = RECT;
    //Resource[18][28] = RECT;

}



void GameMap::SecondMap()
{
    /*
    // 增加资源
    Resource[9][16] = CYCLE;
    Resource[10][16] = CYCLE;
    Resource[10][17] = CYCLE;
    Resource[10][18] = CYCLE;
    Resource[11][17] = CYCLE;
    Resource[11][18] = CYCLE;
    Resource[12][18] = CYCLE;
    Resource[3][5] = RECT;
    Resource[3][6] = RECT;
    Resource[4][6] = RECT;
    Resource[4][7] = RECT;
    Resource[4][8] = RECT;
    Resource[5][7] = RECT;
    Resource[5][6] = RECT;
    Resource[5][5] = RECT;
    Resource[6][5] = RECT;
    Resource[6][8] = RECT;
*/
}

void GameMap::ClearBarriers()
{
    /*
    Resource[1][3] = NONE;
    Resource[2][4] = NONE;
    Resource[4][2] = NONE;
    Resource[5][1] = NONE;
    Resource[9][6] = NONE;
    Resource[10][5] = NONE;
    Resource[9][3] = NONE;
    Resource[12][7] = NONE;
    Resource[8][9] = NONE;
    Resource[12][9] = NONE;
    Resource[12][10] = NONE;
    Resource[13][9] = NONE;
    Resource[13][8] = NONE;
    Resource[14][7] = NONE;
    Resource[2][13] = NONE;
    Resource[2][17] = NONE;
    Resource[3][14] = NONE;
    Resource[3][15] = NONE;
    Resource[4][10] = NONE;
    Resource[4][13] = NONE;
    Resource[5][14] = NONE;
    Resource[5][17] = NONE;
    Resource[6][18] = NONE;
    Resource[6][19] = NONE;
    Resource[7][20] = NONE;
    Resource[8][21] = NONE;
    Resource[9][23] = NONE;
    Resource[9][20] = NONE;
    Resource[10][14] = NONE;
    Resource[11][13] = NONE;
    Resource[12][14] = NONE;
    Resource[14][14] = NONE;
*/
}
int GameMap::GetResource(GridVec pos)
{
    return Resource[pos.i][pos.j];
}
void GameMap::SetBuilding(GridVec pos, Building *building, int direction, int name)
{
    for (auto pos : building->BuildingAllPos())
    {
        BuildingsMap[pos.i][pos.j] = building;
        Buildingsmap[pos.i][pos.j] = building->name;
        build[pos.i*WIDTH+pos.j] = (31 <= name && name <= 42)?name:name * 100 + direction;
    }
}

Building *GameMap::GetBuilding(GridVec pos)
{
    return BuildingsMap[pos.i][pos.j];
}

Building *GameMap::GetBuilding(int i, int j)
{
    return BuildingsMap[i][j];
}

void GameMap::RemoveBuilding(GridVec pos)
{
    if (BuildingsMap[pos.i][pos.j]->name != HUB)
    {
        for (auto pos : BuildingsMap[pos.i][pos.j]->BuildingAllPos())
        {
            BuildingsMap[pos.i][pos.j] = nullptr;
            Buildingsmap[pos.i][pos.j] = -1;
            build[pos.i * WIDTH + pos.j] = -1;
        }
    }
}
GridVec GameMap::GetTatget(GridVec source, int directionout)
{
    GridVec target = source;
    switch (directionout)
    {
        {
        case UP:
            target.i--;
            break;
        case DOWN:
            target.i++;
            break;
        case LEFT:
            target.j--;
            break;
        case RIGHT:
            target.j++;
            break;
        }
    default:
        break;
    }
    return target;
}
int GameMap::OppositeDirection(int direction)
{
    switch (direction)
    {
    case UP:
        return DOWN;
    case DOWN:
        return UP;
    case LEFT:
        return RIGHT;
    case RIGHT:
        return LEFT;
    default:
        return NONE;
    }
}
