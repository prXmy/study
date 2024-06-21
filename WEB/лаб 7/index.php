

<html>
  <head>
    <style>
        .p3{
            color: #FF0000;
        }
        .p2{
           background-color: orange;
        }
        .p1{
            color: green;
        }
    </style>
  </head>
  <body>
  <?php
echo "Hello world";
?>
  <?php
echo "<br> </br>";
$a1 = 0;
echo '<table border = "1">' ;
for ($a1 = 1; $a1 <= 9 ;$a1+=1){
    if($a1<5){
        echo '<tr >';
    }
    else{
    echo '<tr>';
    }
    for ($a2 = 2; $a2 <= 18; $a2+=2){
        if($a2 <10){
            echo '<td class = "p2">';
        }
        else{
        echo '<td >';
        }
        if ($a1%2==1){
            echo'<p class="p1">';
            echo (($a1-1)*18 +$a2);
            echo'</p>';
        echo'</td>';
        }
        else{
        echo'<p >';
        echo(($a1-1)*18 +$a2);
        echo'</p>';
        echo'</td>';
        }
        
    }
    echo'</tr>';
}
echo'</table>';

?>
    </body>
    </html>
