//
//  Voronoi.metal
//  Voronoi tiles
//
//  Created by asd on 21/04/2019.
//  Copyright © 2019 voicesync. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;


typedef uint32_t color; // aa bb gg rr  32 bit color
typedef uint8_t byte;
typedef uint2 point;

inline int sqMag(device point& pnt, int x, int y) {
    int xd = x - pnt.x;
    int yd = y - pnt.y;
    return (xd * xd) + (yd * yd);
}

color genPixel(uint i, uint j, uint count, device point* points, device color*colors) {
    int ind = -1, dist = INT_MAX;
    
    for (uint it = 0; it < count; it++) {
        int d = sqMag(points[it], i, j);
        if (d < dist) {
            dist = d;
            ind = (int)it;
        }
    }
    
    return (ind > -1) ? colors[ind] : 0xff000000;
}

kernel void Voronoi(device color*pixels[[buffer(0)]],
                    device point*points[[buffer(1)]],
                    device color*colors[[buffer(2)]],
                    
                    const device uint &count[[buffer(3)]],  // count
                    
                    uint2 position [[thread_position_in_grid]],
                    uint2 tpg[[threads_per_grid]])
{
    uint x=position.x, y=position.y, width=tpg.x;
    pixels[x + y * width] = genPixel(x, y, count, points, colors);
}

// set inner black pixel
kernel void setPointBox(device color*pixels[[buffer(0)]],
                      device point*points[[buffer(1)]],
                      const device uint &width[[buffer(2)]],   // width
                      
                      uint2 position [[thread_position_in_grid]]
                      )
{
   int i=position.x;
   int x = points[i].x, y = points[i].y;
   const color black=0xff000000;
   
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2; j++)
          pixels[(x + i) + width * (y + j)] = black;
}
