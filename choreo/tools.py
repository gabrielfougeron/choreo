"""

"""


import os
import sys
import shutil
import math

try:
    import ffmpeg
except:
    pass

 
def factor_squarest(n):
    x = math.ceil(math.sqrt(n))
    y = int(n/x)
    while ( (y * x) != float(n) ):
        x -= 1
        y = int(n/x)
    return max(x,y), min(x,y)

def VideoGrid(input_list, output_filename, nxy = None, ordering='RowMajor'):

    nvid = len(input_list)

    if nxy is None:
         nx,ny = factor_squarest(nvid)

    else:
        nx,ny = nxy
        if (nx*ny != nvid):
            raise(ValueError('The number of input video files is incorrect'))

    if nvid == 1:
        
        shutil.copy(input_list[0], output_filename)

    else:
        
        if ordering == 'RowMajor':
            layout_list = []
            for iy in range(ny):
                if iy == 0:
                    ylayout='0'
                else:
                    ylayout = 'h0'
                    for iiy in range(1,iy):
                        ylayout = ylayout + '+h'+str(iiy)

                for ix in range(nx):
                    if ix == 0:
                        xlayout='0'
                    else:
                        xlayout = 'w0'
                        for iix in range(1,ix):
                            xlayout = xlayout + '+w'+str(iix)


                    layout_list.append(xlayout + '_' + ylayout)

        elif ordering == 'ColMajor':
            layout_list = []
            for ix in range(nx):
                if ix == 0:
                    xlayout='0'
                else:
                    xlayout = 'w0'
                    for iix in range(1,ix):
                        xlayout = xlayout + '+w'+str(iix)
                for iy in range(ny):
                    if iy == 0:
                        ylayout='0'
                    else:
                        ylayout = 'h0'
                        for iiy in range(1,iy):
                            ylayout = ylayout + '+h'+str(iiy)

                    layout_list.append(xlayout + '_' + ylayout)
        else:
            raise(ValueError('Unknown ordering : '+ordering))

        layout = layout_list[0]
        for i in range(1,nvid):
            layout = layout + '|' + layout_list[i]

        try:
            
            ffmpeg_input_list = []
            for the_input in input_list:
                ffmpeg_input_list.append(ffmpeg.input(the_input))

            # ffmpeg_out = ( ffmpeg
            #     .filter(ffmpeg_input_list, 'hstack')
            # )
    # 
            ffmpeg_out = ( ffmpeg
                .filter(
                    ffmpeg_input_list,
                    'xstack',
                    inputs=nvid,
                    layout=layout,
                )
            )

            ffmpeg_out = ( ffmpeg_out
                .output(output_filename,vcodec='h264',pix_fmt='yuv420p')
                .global_args('-y')
                .global_args('-loglevel','error')
            )

            ffmpeg_out.run()

        # except:
        #     raise ModuleNotFoundError('Error: ffmpeg not found')

        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
