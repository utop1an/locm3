;; rows=7, columns=3, robots=1, out_folder=training/easy, instance_id=38, seed=63

(define (problem floortile-38)
 (:domain floortile)
 (:objects 
    tile_0_1
    tile_0_2
    tile_0_3
    tile_1_1
    tile_1_2
    tile_1_3
    tile_2_1
    tile_2_2
    tile_2_3
    tile_3_1
    tile_3_2
    tile_3_3
    tile_4_1
    tile_4_2
    tile_4_3
    tile_5_1
    tile_5_2
    tile_5_3
    tile_6_1
    tile_6_2
    tile_6_3
    tile_7_1
    tile_7_2
    tile_7_3 - tile
    robot1 - robot
    white black - color
)
 (:init 
    (robot-at robot1 tile_4_1)
    (robot-has robot1 black)
    (available-color white)
    (available-color black)
    (clear tile_0_1)
    (clear tile_0_2)
    (clear tile_0_3)
    (clear tile_1_1)
    (clear tile_1_2)
    (clear tile_1_3)
    (clear tile_2_1)
    (clear tile_2_2)
    (clear tile_2_3)
    (clear tile_3_1)
    (clear tile_3_2)
    (clear tile_3_3)
    (clear tile_4_2)
    (clear tile_4_3)
    (clear tile_5_1)
    (clear tile_5_2)
    (clear tile_5_3)
    (clear tile_6_1)
    (clear tile_6_2)
    (clear tile_6_3)
    (clear tile_7_1)
    (clear tile_7_2)
    (clear tile_7_3)
    (up tile_1_1 tile_0_1 )
    (up tile_1_2 tile_0_2 )
    (up tile_1_3 tile_0_3 )
    (up tile_2_1 tile_1_1 )
    (up tile_2_2 tile_1_2 )
    (up tile_2_3 tile_1_3 )
    (up tile_3_1 tile_2_1 )
    (up tile_3_2 tile_2_2 )
    (up tile_3_3 tile_2_3 )
    (up tile_4_1 tile_3_1 )
    (up tile_4_2 tile_3_2 )
    (up tile_4_3 tile_3_3 )
    (up tile_5_1 tile_4_1 )
    (up tile_5_2 tile_4_2 )
    (up tile_5_3 tile_4_3 )
    (up tile_6_1 tile_5_1 )
    (up tile_6_2 tile_5_2 )
    (up tile_6_3 tile_5_3 )
    (up tile_7_1 tile_6_1 )
    (up tile_7_2 tile_6_2 )
    (up tile_7_3 tile_6_3 )
    (down tile_0_1 tile_1_1 )
    (down tile_0_2 tile_1_2 )
    (down tile_0_3 tile_1_3 )
    (down tile_1_1 tile_2_1 )
    (down tile_1_2 tile_2_2 )
    (down tile_1_3 tile_2_3 )
    (down tile_2_1 tile_3_1 )
    (down tile_2_2 tile_3_2 )
    (down tile_2_3 tile_3_3 )
    (down tile_3_1 tile_4_1 )
    (down tile_3_2 tile_4_2 )
    (down tile_3_3 tile_4_3 )
    (down tile_4_1 tile_5_1 )
    (down tile_4_2 tile_5_2 )
    (down tile_4_3 tile_5_3 )
    (down tile_5_1 tile_6_1 )
    (down tile_5_2 tile_6_2 )
    (down tile_5_3 tile_6_3 )
    (down tile_6_1 tile_7_1 )
    (down tile_6_2 tile_7_2 )
    (down tile_6_3 tile_7_3 )
    (left tile_0_1 tile_0_2 )
    (left tile_0_2 tile_0_3 )
    (left tile_1_1 tile_1_2 )
    (left tile_1_2 tile_1_3 )
    (left tile_2_1 tile_2_2 )
    (left tile_2_2 tile_2_3 )
    (left tile_3_1 tile_3_2 )
    (left tile_3_2 tile_3_3 )
    (left tile_4_1 tile_4_2 )
    (left tile_4_2 tile_4_3 )
    (left tile_5_1 tile_5_2 )
    (left tile_5_2 tile_5_3 )
    (left tile_6_1 tile_6_2 )
    (left tile_6_2 tile_6_3 )
    (left tile_7_1 tile_7_2 )
    (left tile_7_2 tile_7_3 )
    (right tile_0_2 tile_0_1 )
    (right tile_0_3 tile_0_2 )
    (right tile_1_2 tile_1_1 )
    (right tile_1_3 tile_1_2 )
    (right tile_2_2 tile_2_1 )
    (right tile_2_3 tile_2_2 )
    (right tile_3_2 tile_3_1 )
    (right tile_3_3 tile_3_2 )
    (right tile_4_2 tile_4_1 )
    (right tile_4_3 tile_4_2 )
    (right tile_5_2 tile_5_1 )
    (right tile_5_3 tile_5_2 )
    (right tile_6_2 tile_6_1 )
    (right tile_6_3 tile_6_2 )
    (right tile_7_2 tile_7_1 )
    (right tile_7_3 tile_7_2 ))
 (:goal  (and 
    (painted tile_1_1 white)
    (painted tile_1_2 black)
    (painted tile_1_3 white)
    (painted tile_2_1 black)
    (painted tile_2_2 white)
    (painted tile_2_3 black)
    (painted tile_3_1 white)
    (painted tile_3_2 black)
    (painted tile_3_3 white)
    (painted tile_4_1 black)
    (painted tile_4_2 white)
    (painted tile_4_3 black)
    (painted tile_5_1 white)
    (painted tile_5_2 black)
    (painted tile_5_3 white)
    (painted tile_6_1 black)
    (painted tile_6_2 white)
    (painted tile_6_3 black)
    (painted tile_7_1 white)
    (painted tile_7_2 black)
    (painted tile_7_3 white))))
