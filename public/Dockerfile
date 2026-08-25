FROM nginx:alpine
# Copy all contents of your public folder into Nginx's web root
COPY ./ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
