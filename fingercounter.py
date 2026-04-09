import mediapipe as mp
import cv2

mp_maos = mp.solutions.hands

maos = mp_maos.Hands(
    max_num_hands=4,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

imagem = cv2.imread("absolutecinema.jpg")

mp_desenho = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pontos_dedos = [4,8,12,16,20]


while True:
    red, frame = cap.read()
    if not red:
        break
    frame = cv2.resize(frame, (1220, 720))
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = maos.process(frame_rgb)
    
    dedos_esquerda=0
    dedos_direita=0
    if resultado.multi_hand_landmarks:

         for idx, hand in enumerate(resultado.multi_hand_landmarks):

            mp_desenho.draw_landmarks(frame, hand, mp_maos.HAND_CONNECTIONS)
            pontos = hand.landmark

           
            tipo = resultado.multi_handedness[idx].classification[0].label
            if tipo == "Right":
                if pontos[5].x < pontos[17].x:
                    if pontos[pontos_dedos[0]].x < pontos[pontos_dedos[0] - 1].x:
                        dedos_direita +=1
                else:
                    if pontos[pontos_dedos[0]].x > pontos[pontos_dedos[0] - 1].x:
                        dedos_direita +=1

                for i in range(1, 5):
                    if pontos[1].y < pontos[12].y:
                        if pontos[pontos_dedos[i]].y > pontos[pontos_dedos[i] - 2].y:
                            dedos_direita +=1
                    else:
                        if pontos[pontos_dedos[i]].y < pontos[pontos_dedos[i] - 2].y:
                            dedos_direita +=1
                            total = dedos_direita+dedos_esquerda
                            if total == 10: #PRA DESATIVAR A IMAGEM APARECENDO BASTA APAGAR ESSA CONDIÇÃO
                                frame[-230:-30, -750:-550] = cv2.resize(imagem, (200, 200))
                            
            if tipo == "Left":
                if pontos[5].x > pontos[17].x:
                    if pontos[pontos_dedos[0]].x > pontos[pontos_dedos[0] - 1].x:
                        dedos_esquerda +=1
                else:
                    if pontos[pontos_dedos[0]].x < pontos[pontos_dedos[0] - 1].x:
                        dedos_esquerda +=1

                for i in range(1, 5):
                    if pontos[1].y < pontos[12].y:
                        if pontos[pontos_dedos[i]].y > pontos[pontos_dedos[i] - 2].y:
                            dedos_esquerda +=1
                    else:
                        if pontos[pontos_dedos[i]].y < pontos[pontos_dedos[i] - 2].y:
                            dedos_esquerda +=1
                            total = dedos_direita+dedos_esquerda
                            if total == 10: #PRA DESATIVAR A IMAGEM APARECENDO BASTA APAGAR ESSA CONDIÇÃO TAMBÉM
                                frame[-230:-30, -750:-550] = cv2.resize(imagem, (200, 200))
            
                        
    
    
    cv2.putText(frame, f" Total: {dedos_esquerda+dedos_direita}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    cv2.putText(frame, f" Esquerda: {dedos_esquerda}" if dedos_esquerda!=0 else "", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.putText(frame, f"Direita: {dedos_direita}" if dedos_direita !=0 else "", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow("FingerCounter - AI Vision", frame)
    
    tecla = cv2.waitKey(1)
    if tecla == 27:
        print("Saindo")
        break
    
cap.release()
cv2.destroyAllWindows()
