Konfiguracja google cloud.	

Uruchomienie jupytera w chmurze.	

1. Rejestracja w google cloud	
2. Stworzenie maszyny wirtualnej (Computer Engine -> Create VM Instance {nazwa, lokalizacja, CPU, memory, system })	
3. External IP (adres naszego serwera) i logujemy się przez console Connect Open in browser windows	
4. Instalacja Anacondy.	
5. Ustawienie firewall w google cloud (firewall rules)	
6. jupyter notebook --generate-config 	
  (otworzyc stworzony plik zmienic 2 rzeczy:     	
     odkomentować c.NotebookApp.allow_orgin = '*'    	
     odkomentować c.NotebookApp.ipconfig = '0.0.0.0' )	
   Zapisać zmiany.	
7. jupyter notebook - skopiować token	
8. Teraz można odpalić jupytera w google cloud {External IP}:8888 i hasło jako token :-)